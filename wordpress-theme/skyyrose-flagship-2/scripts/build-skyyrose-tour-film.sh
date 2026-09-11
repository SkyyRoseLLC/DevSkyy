#!/usr/bin/env bash
set -euo pipefail

# Eight approved jersey-front photographs in the fictional SkyyRose Town Line tour.
# Photographs remain intact; chapter timing is shared with the scroll-world registry.
theme_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$theme_dir/assets/video"
source_frame="${1:?Usage: build-skyyrose-tour-film.sh <tracked-reference-frame> <tracked-on-model-directory>}"
model_dir="${2:?Usage: build-skyyrose-tour-film.sh <tracked-reference-frame> <tracked-on-model-directory>}"
proof_dir="${3:-$theme_dir/../../.artifacts/front-card-release-20260905}"
font_ui="/System/Library/Fonts/Avenir Next Condensed.ttc"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required" >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ffprobe is required" >&2; exit 1; }
command -v magick >/dev/null 2>&1 || { echo "ImageMagick is required" >&2; exit 1; }

models=(
	"$model_dir/br-003-onmodel.webp"
	"$model_dir/br-008-onmodel.webp"
	"$model_dir/br-009-onmodel.webp"
	"$model_dir/br-010-onmodel.webp"
	"$model_dir/br-011-onmodel.webp"
	"$model_dir/br-012-onmodel.webp"
	"$model_dir/br-014-onmodel.webp"
	"$model_dir/br-015-onmodel.webp"
)
for required in "$source_frame" "$font_ui" "${models[@]}"; do
	[[ -f "$required" ]] || { echo "Missing required source: $required" >&2; exit 1; }
done

mkdir -p "$output_dir" "$proof_dir"
shasum -a 256 "${models[@]}" > "$proof_dir/film-source-sha256.txt"
render_dir="$(mktemp -d "${TMPDIR:-/tmp}/skyyrose-tour-film.XXXXXX")"
trap 'rm -rf "$render_dir"' EXIT

titles=(
	"00 / BASEBALL CLASSIC"
	"01 / SF INSPIRED"
	"02 / LAST OAKLAND"
	"03 / THE BAY"
	"04 / THE ROSE"
	"05 / OAKLAND GREEN"
	"06 / SAN FRANCISCO ORANGE"
	"07 / BASEBALL WHITE"
)
chapters=( "FOUNDATION" "SAN FRANCISCO" "OAKLAND" "THE BAY" "SAN JOSE" "OAKLAND GREEN" "SAN FRANCISCO" "HOME" )
skus=( br-003 br-008 br-009 br-010 br-011 br-012 br-014 br-015 )

magick "$source_frame" \
	-resize '1920x1080^' -gravity center -extent 1920x1080 \
	-colorspace sRGB -blur 0x5 -brightness-contrast -28x12 "$render_dir/background.png"

for index in 0 1 2 3 4 5 6 7; do
	frame="$render_dir/frame-$index.png"
	magick "$render_dir/background.png" \
		-fill '#00000080' -draw 'rectangle 0,0 1920,1080' \
		-fill '#050505b8' -stroke '#c9a85e' -strokewidth 2 -draw 'rectangle 88,86 1128,856' \
		-stroke '#c9a85e' -strokewidth 4 -draw 'line 88,86 88,856' \
		\( "${models[$index]}" -resize '650x975>' \) -gravity northeast -geometry +65+52 -composite -gravity northwest -stroke none \
		-font "$font_ui" -fill '#c9a85e' -pointsize 28 -annotate +132+132 'SKYY ROSE / JERSEY SERIES' \
		-font "$font_ui" -fill white -pointsize 64 -annotate +132+245 'TOUR AROUND THE BAY' \
		-font "$font_ui" -fill '#d6d0c7' -pointsize 24 -annotate +132+315 'OAKLAND  —  SAN FRANCISCO  —  THE BAY  —  SAN JOSE' \
		-fill white -pointsize 44 -annotate +132+430 "${titles[$index]}" \
		-font "$font_ui" -fill '#d6d0c7' -pointsize 34 -annotate +132+525 'Carry every number home.' \
		-font "$font_ui" -fill '#c9a85e' -pointsize 22 -annotate +132+635 'BLACK IS BEAUTIFUL' \
		-stroke '#c9a85e' -strokewidth 3 -draw 'line 150,770 1010,770' \
		-fill '#c9a85e' -stroke none -draw 'circle 160,770 172,770 circle 370,770 382,770 circle 580,770 592,770 circle 790,770 802,770 circle 1000,770 1012,770' \
		-font "$font_ui" -fill white -pointsize 19 \
		-annotate +132+826 'FOUNDATION' -annotate +338+826 'SF' -annotate +536+826 'OAKLAND' -annotate +748+826 'THE BAY' -annotate +948+826 'SAN JOSE' \
		-fill '#c9a85e' -pointsize 24 -annotate +132+900 "TOUR CHAPTER / ${chapters[$index]}" "$frame"
	cp "$frame" "$proof_dir/${skus[$index]}-film-frame.png"
done

inputs=()
filters=""
for index in 0 1 2 3 4 5 6 7; do
	inputs+=( -loop 1 -t 3.6 -i "$render_dir/frame-$index.png" )
	filters+="[$index:v]fps=30,format=yuv420p[v$index];"
done
previous="v0"
for index in 1 2 3 4 5 6 7; do
	offset="$(awk -v chapter="$index" 'BEGIN { printf "%.1f", chapter * 3.1 }')"
	filters+="[$previous][v$index]xfade=transition=fade:duration=0.5:offset=$offset[x$index];"
	previous="x$index"
done
filters+="[$previous]fade=t=in:st=0:d=0.45,fade=t=out:st=24.5:d=0.8,format=yuv420p[outv]"
ffmpeg -hide_banner -loglevel error -y "${inputs[@]}" \
	-filter_complex_threads 2 -filter_complex "$filters" \
	-map "[outv]" -an -r 30 -t 25.3 -c:v libx264 -preset slow -crf 20 -movflags +faststart \
	"$output_dir/skyyrose-tour-around-the-bay.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$output_dir/skyyrose-tour-around-the-bay.mp4" -an \
	-c:v libvpx-vp9 -crf 34 -b:v 0 -row-mt 1 "$output_dir/skyyrose-tour-around-the-bay.webm"
ffmpeg -hide_banner -loglevel error -y -ss 1.2 -i "$output_dir/skyyrose-tour-around-the-bay.mp4" -frames:v 1 \
	-vf "scale=1280:-2" "$render_dir/poster.png"
magick "$render_dir/poster.png" -quality 86 "$output_dir/skyyrose-tour-around-the-bay-poster.webp"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate -of default=noprint_wrappers=1 "$output_dir/skyyrose-tour-around-the-bay.mp4"
ffprobe -v error -show_format -show_streams -of json "$output_dir/skyyrose-tour-around-the-bay.mp4" > "$proof_dir/film-ffprobe.json"
for index in 0 1 2 3 4 5 6 7; do
	proof_time="$(awk -v chapter="$index" 'BEGIN { printf "%.1f", chapter * 3.1 + 1.2 }')"
	ffmpeg -hide_banner -loglevel error -y -ss "$proof_time" -i "$output_dir/skyyrose-tour-around-the-bay.mp4" -frames:v 1 "$proof_dir/${skus[$index]}-film-frame.png"
done
magick montage -font "$font_ui" "$proof_dir"/br-*-film-frame.png -thumbnail 480x270 -tile 2x4 -geometry +8+8 -background '#111111' "$proof_dir/jersey-film-contact-sheet.jpg"
