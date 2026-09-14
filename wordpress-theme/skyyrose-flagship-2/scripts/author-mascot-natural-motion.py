"""Author contact-driven gait and a single, phased greeting on the existing rig."""

import json
import math
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

if not bpy.data.filepath:
    raise RuntimeError("Open the fitted Skyy blend first.")
root = Path(bpy.data.filepath).resolve().parent
scene = bpy.data.scenes["Skyy Candidate Studio"]
bpy.context.window.scene = scene
scene.render.fps = 30
rig = bpy.data.objects["SkyyCandidateRig"]
mesh = bpy.data.objects["SkyyCandidateMesh"]
selected = globals().get("REAUTHOR_CLIPS")
rig.animation_data.action = None
for track in list(rig.animation_data.nla_tracks):
    if selected is None or track.name.removeprefix("Skyy_") in selected:
        rig.animation_data.nla_tracks.remove(track)
for action in list(bpy.data.actions):
    if action.name.startswith("Natural_Skyy_") and (
        selected is None or action.name.removeprefix("Natural_Skyy_") in selected
    ):
        bpy.data.actions.remove(action)
rest = {b.name: b.matrix_local.copy() for b in rig.data.bones}
heads = {b.name: b.head_local.copy() for b in rig.data.bones}
sole = {
    s: [
        v.co.copy() - heads["foot." + s]
        for v in mesh.data.vertices
        if v.co.z < 0.025 and (v.co.x > 0) == (s == "L")
    ]
    for s in ["L", "R"]
}
clamps = []


def ease(x):
    x = max(0, min(1, x))
    return max(0, min(1, x * x * x * (x * (x * 6 - 15) + 10)))


def lerp(a, b, t):
    return a + (b - a) * t


def world_bone(name, head, delta):
    rig.pose.bones[name].matrix = (
        Matrix.Translation(head) @ (delta.to_matrix() @ rest[name].to_3x3()).to_4x4()
    )
    bpy.context.view_layer.update()


def body(name, head, angles):
    world_bone(name, head, Euler(tuple(math.radians(v) for v in angles), "XYZ").to_quaternion())


def root_point(name):
    b = rig.pose.bones[name]
    return (
        b.parent.matrix @ rest[b.parent.name].inverted() @ heads[name]
        if b.parent
        else heads[name].copy()
    )


def aim(name, head, tail):
    direction = tail - head
    rotation = (rig.data.bones[name].tail_local - heads[name]).rotation_difference(direction)
    world_bone(name, head, rotation)


def limb(upper, lower, end, pole):
    start = root_point(upper)
    a = rig.data.bones[upper].length
    b = rig.data.bones[lower].length
    direction = end - start
    distance = direction.length
    if distance > a + b - 0.0001:
        clamps.append({"bone": upper, "excess": distance - (a + b - 0.0001)})
        end = start + direction.normalized() * (a + b - 0.0001)
        direction = end - start
        distance = direction.length
    n = direction.normalized()
    along = (a * a - b * b + distance * distance) / (2 * distance)
    radius = math.sqrt(max(0, a * a - along * along))
    p = pole - start
    p -= n * p.dot(n)
    joint = start + n * along + p.normalized() * radius
    aim(upper, start, joint)
    aim(lower, joint, end)
    return end


def set_pose(clip, time, duration):
    phase = time / duration
    tau = phase * math.tau
    for b in rig.pose.bones:
        b.matrix_basis = Matrix.Identity(4)
        b.rotation_mode = "QUATERNION"
    gait = clip in ("Walk", "Exit")
    greet = clip == "Wave"
    joypulse = math.sin(math.pi * phase) ** 2 if clip == "Joy" else 0
    env = ease(time / 0.8) * (1 - ease((time - 1.95) / 0.85)) if greet else 0
    weight = 0.009 * math.sin(tau) if gait else (0.008 * env if greet else 0.0025 * math.sin(tau))
    vertical = (
        -0.013 + 0.004 * (1 - math.cos(2 * tau)) / 2 if gait else -0.004 * env - 0.006 * joypulse
    )
    breath = 0.30 * math.sin(tau)
    pelvis_angles = (1.8, 1.1 * math.sin(tau), -2.8 * math.cos(tau)) if gait else (0, 0, 1.0 * env)
    body("pelvis", heads["pelvis"] + Vector((weight, 0, vertical)), pelvis_angles)
    body(
        "spine",
        rig.pose.bones["pelvis"].tail,
        (1.0 if gait else breath, 0, 1.3 * math.cos(tau) if gait else -0.8 * env),
    )
    body(
        "chest",
        rig.pose.bones["spine"].tail,
        (
            1.8 if gait else breath - 1.8 * joypulse,
            -0.8 * math.sin(tau) if gait else -0.8 * env,
            2 * math.cos(tau) if gait else -1.5 * env,
        ),
    )
    body("neck", rig.pose.bones["chest"].tail, (0.3, 0, 0))
    head_lead = ease(time / 0.48) * (1 - ease((time - 2.05) / 0.75)) if greet else 0
    nod = (
        1.6 * math.sin(tau * 2) * math.sin(math.pi * phase) ** 2
        if clip == "Talk"
        else 0.5 * math.sin(tau)
    )
    body(
        "head",
        rig.pose.bones["neck"].tail,
        (
            0.5 * math.sin(tau - 0.3) if gait else nod - 2.5 * joypulse,
            -2.5 * head_lead,
            -4 * head_lead + 2 * joypulse,
        ),
    )
    for side, sign in [("L", 1), ("R", -1)]:
        if gait:
            q = (phase + (0 if side == "L" else 0.5)) % 1
            stance = 0.62
            stride = 0.15
            if q < stance:
                foot_y = -stride / 2 + stride * q / stance
                lift = 0
                pitch = -9 * (1 - ease(q / 0.10)) + 17 * ease((q - 0.48) / (0.62 - 0.48))
            else:
                u = (q - stance) / (1 - stance)
                tangent = stride * (1 - stance) / stance
                foot_y = (
                    (2 * u**3 - 3 * u * u + 1) * (stride / 2)
                    + (u**3 - 2 * u * u + u) * tangent
                    + (-2 * u**3 + 3 * u * u) * (-stride / 2)
                    + (u**3 - u * u) * tangent
                )
                lift = 0.034 * math.sin(math.pi * u) ** 1.4
                pitch = lerp(17, -9, ease(u))
        else:
            foot_y = 0
            lift = 0
            pitch = 0
        rot = Euler((math.radians(pitch), 0, 0)).to_quaternion()
        ankle_z = 0.002 + lift - min((rot @ p).z for p in sole[side])
        ankle = Vector((sign * 0.13, foot_y, ankle_z))
        ankle = limb("thigh." + side, "shin." + side, ankle, Vector((sign * 0.12, -0.30, 0.245)))
        world_bone("foot." + side, ankle, rot)
    for side, sign in [("L", 1), ("R", -1)]:
        clav = "clavicle." + side
        chest_delta = (
            rig.pose.bones["chest"].matrix.to_quaternion()
            @ rest["chest"].to_quaternion().inverted()
        )
        lift_angle = math.radians(-sign * 5 * env if greet and side == "R" else 0)
        world_bone(clav, root_point(clav), chest_delta @ Quaternion((0, 1, 0), lift_angle))
        wrist = Vector((sign * 0.14 + weight * 0.6, -0.025, 0.405 + vertical * 0.5))
        pole = Vector((sign * 0.125, 0.14, 0.52 + vertical * 0.5))
        if gait:
            wrist.y += sign * 0.032 * math.cos(tau - 0.25)
            wrist.z += 0.005 * (1 - math.cos(tau))
        if greet and side == "R":
            # Single curved lift and return: no artificial stop at a mid-air waypoint.
            down = Vector((-0.14, -0.025, 0.405))
            prep = Vector((-0.147, 0.004, 0.408))
            up = Vector((-0.22, -0.075, 0.715))
            if time < 0.24:
                wrist = down.lerp(prep, ease(time / 0.24))
            elif time < 1.10:
                u = ease((time - 0.24) / 0.86)
                wrist = prep.lerp(up, u)
                wrist.x -= 0.026 * math.sin(math.pi * u)
                wrist.y -= 0.017 * math.sin(math.pi * u)
            elif time < 1.85:
                wrist = up.copy()
            else:
                u = ease((time - 1.85) / 0.95)
                wrist = up.lerp(down, u)
                wrist.x -= 0.02 * math.sin(math.pi * u)
                wrist.y += 0.025 * math.sin(math.pi * u)
            pulse = (
                math.sin((time - 0.9) * math.tau / 0.62)
                * ease((time - 0.8) / 0.2)
                * (1 - ease((time - 1.65) / 0.3))
            )
            wrist.x += 0.008 * pulse
            wrist.z += 0.003 * pulse
            pole = pole.lerp(Vector((-0.23, 0.12, 0.50)), env)
        elif clip == "Talk":
            envtalk = math.sin(math.pi * phase) ** 2 * (
                0.72 + 0.28 * math.sin(tau + (0 if side == "L" else 1.4))
            )
            wrist.y -= 0.026 * envtalk
            wrist.z += (0.034 if side == "L" else 0.019) * envtalk
            wrist.x += sign * 0.008 * math.sin(tau) * envtalk
        elif clip == "Joy":
            joy = math.sin(math.pi * phase) ** 2
            wrist.z += (0.045 if side == "L" else 0.035) * joy
            wrist.y -= 0.025 * joy
        wrist = limb("upper_arm." + side, "forearm." + side, wrist, pole)
        hand = rig.pose.bones["hand." + side]
        inherited = (
            rig.pose.bones["forearm." + side].matrix.to_quaternion()
            @ rest["forearm." + side].to_quaternion().inverted()
            @ rest["hand." + side].to_quaternion()
        )
        if greet and side == "R":
            fan = (
                math.radians(11)
                * math.sin((time - 0.9) * math.tau / 0.62)
                * ease((time - 0.8) / 0.2)
                * (1 - ease((time - 1.65) / 0.3))
            )
            # Finger direction up, palm toward the visitor. Wrist leads the small wave.
            greeting = (
                Quaternion((0, 1, 0), fan)
                @ Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))).to_quaternion()
            )
            palm_env = ease((time - 0.30) / 0.80) * (1 - ease((time - 1.88) / 0.90))
            orientation = inherited.slerp(greeting, palm_env)
        else:
            orientation = inherited
            if gait:
                orientation = (
                    Quaternion((1, 0, 0), math.radians(sign * 4 * math.sin(tau - 0.7)))
                    @ orientation
                )
        hand.matrix = Matrix.Translation(wrist) @ orientation.to_matrix().to_4x4()
        bpy.context.view_layer.update()
        # Intermediate rotations share each anatomical pivot and reduce
        # linear-skinning collapse without Blender-only dual quaternion skinning.
        up = rig.pose.bones["upper_arm." + side]
        fore = rig.pose.bones["forearm." + side]
        uq = up.matrix.to_quaternion() @ rest[up.name].to_quaternion().inverted()
        fq = fore.matrix.to_quaternion() @ rest[fore.name].to_quaternion().inverted()
        hq = orientation @ rest[hand.name].to_quaternion().inverted()
        world_bone("shoulder_blend." + side, up.head, chest_delta.slerp(uq, 0.5))
        world_bone("elbow_blend." + side, fore.head, uq.slerp(fq, 0.5))
        world_bone("wrist_blend." + side, hand.head, fq.slerp(hq, 0.5))
        # Relax the finger chain in locomotion; open as the greeting rises.
        opening = (
            ease((time - 0.25) / 0.83) * (1 - ease((time - 1.90) / 0.80))
            if greet and side == "R"
            else 0
        )
        curl = (15 + 3 * math.sin(tau - 0.7) if gait else 15) * (1 - opening)
        tipcurl = (12 + 2 * math.sin(tau - 0.9) if gait else 12) * (1 - opening)
        if clip == "Talk":
            curl *= 1 - 0.35 * envtalk
            tipcurl *= 1 - 0.35 * envtalk
        gripq = hq @ Quaternion((0, 1, 0), math.radians(sign * curl))
        world_bone("grip." + side, root_point("grip." + side), gripq)
        world_bone(
            "fingertips." + side,
            root_point("fingertips." + side),
            gripq @ Quaternion((0, 1, 0), math.radians(sign * tipcurl)),
        )


specs = [("Idle", 4), ("Walk", 1.1), ("Wave", 2.8), ("Talk", 3.2), ("Joy", 1.5), ("Exit", 1.1)]
for clip, duration in specs:
    if selected is not None and clip not in selected:
        continue
    action = bpy.data.actions.new("Natural_Skyy_" + clip)
    rig.animation_data.action = action
    count = round(duration * 30)
    for frame in range(count + 1):
        scene.frame_set(frame + 1)
        set_pose(clip, frame / 30, duration)
        for bone in rig.pose.bones:
            if bone.name == "neutral_bone":
                continue
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame + 1, group=bone.name)
            bone.keyframe_insert(data_path="location", frame=frame + 1, group=bone.name)
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for p in curve.keyframe_points:
                        p.interpolation = "LINEAR"
    track = rig.animation_data.nla_tracks.new()
    track.name = "Skyy_" + clip
    track.strips.new("Skyy_" + clip, 1, action)
    track.mute = True
rig.animation_data.action = bpy.data.actions["Natural_Skyy_Idle"]
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(root / "skyy-natural-working.blend"))
generated_specs = [item for item in specs if selected is None or item[0] in selected]
(root / "natural-motion-authoring.json").write_text(
    json.dumps({"clips": generated_specs, "reach_clamps": clamps}, indent=2)
)
result = {
    "clips": generated_specs,
    "reach_clamp_count": len(clamps),
    "worst_reach_excess": max((c["excess"] for c in clamps), default=0),
}
