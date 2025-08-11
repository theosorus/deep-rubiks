// roll.js
import * as THREE from "three";

export class Roll {
  constructor({faceCond, direction, scene, pivot, cubes, steps, stepAngle}){
    this.face = faceCond; this.dir = direction;
    this.scene = scene; this.pivot = pivot; this.cubes = cubes;
    this.steps = steps; this.stepAngle = stepAngle;
    this.stepCount = 0; this.active = true;
    this._pickLayer();
  }
  _pickLayer(){
    this.cubes.forEach((m)=>{
      if (m.position[this.face.axis] === this.face.value){
        this.scene.remove(m); this.pivot.add(m);
      }
    });
  }
  rollFace(){
    if (this.stepCount < this.steps){
      this.pivot.rotation[this.face.axis] += this.dir * this.stepAngle;
      this.stepCount++; return;
    }
    if (!this.active) return;
    this.active = false; this._bake();
  }
  _bake(){
    const q = new THREE.Quaternion();
    this.pivot.updateWorldMatrix(true, true);
    for (let i=this.pivot.children.length-1; i>=0; i--){
      const m = this.pivot.children[i];
      m.updateWorldMatrix(true,false);
      m.getWorldPosition(m.position);
      m.getWorldQuaternion(q);
      m.quaternion.copy(q);
      m.position.x = Math.round(m.position.x);
      m.position.y = Math.round(m.position.y);
      m.position.z = Math.round(m.position.z);
      this.pivot.remove(m); this.scene.add(m);
    }
    this.pivot.rotation.set(0,0,0);
  }
}