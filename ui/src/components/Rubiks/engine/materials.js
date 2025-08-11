// materials.js
import * as THREE from "three";

const vertexShader = `
  varying vec2 vUv;
  void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }
`;
const fragmentShader = `
  varying vec2 vUv; uniform vec3 faceColor;
  void main(){
    vec3 border = vec3(0.533);
    float bl=smoothstep(0.0,0.03,vUv.x);
    float br=smoothstep(1.0,0.97,vUv.x);
    float bt=smoothstep(0.0,0.03,vUv.y);
    float bb=smoothstep(1.0,0.97,vUv.y);
    vec3 c = mix(border, faceColor, bt*br*bb*bl);
    gl_FragColor = vec4(c,1.0);
  }
`;

const makeMat = (hex) =>
  new THREE.ShaderMaterial({
    vertexShader, fragmentShader,
    uniforms: { faceColor: { value: new THREE.Color(hex) } },
  });

export const createMaterials = () => ({
  blue: makeMat("#0000FF"),
  red: makeMat("#FF0000"),
  white: makeMat("#FFFFFF"),
  green: makeMat("#009B48"),
  yellow: makeMat("#FFFF00"),
  orange: makeMat("#FFA500"),
  gray: makeMat("#777777"),
});