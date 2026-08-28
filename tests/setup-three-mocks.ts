import threeMock from './__mocks__/three.cjs';

if (!threeMock.CanvasTexture) {
  throw new Error('three.cjs mock is missing CanvasTexture export');
}

Object.assign(globalThis, { THREE: threeMock });
