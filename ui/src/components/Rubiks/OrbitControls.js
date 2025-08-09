import * as THREE from 'three';
 
// OrbitControls implementation (simplified version)
class OrbitControls {
    constructor(camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.enableRotate = true;
    this.rotateSpeed = 1.0;
    this.enableZoom = true;
    this.zoomSpeed = 1.0;
    this.enablePan = true;
    this.keyPanSpeed = 7.0;
    this.autoRotate = false;
    this.autoRotateSpeed = 2.0;

    this.spherical = new THREE.Spherical();
    this.sphericalDelta = new THREE.Spherical();
    this.scale = 1;
    this.panOffset = new THREE.Vector3();
    this.zoomChanged = false;

    this.rotateStart = new THREE.Vector2();
    this.rotateEnd = new THREE.Vector2();
    this.rotateDelta = new THREE.Vector2();

    this.panStart = new THREE.Vector2();
    this.panEnd = new THREE.Vector2();
    this.panDelta = new THREE.Vector2();

    this.dollyStart = new THREE.Vector2();
    this.dollyEnd = new THREE.Vector2();
    this.dollyDelta = new THREE.Vector2();

    this.target = new THREE.Vector3();

    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseWheel = this.onMouseWheel.bind(this);

    this.domElement.addEventListener('mousedown', this.onMouseDown);
    this.domElement.addEventListener('wheel', this.onMouseWheel);
    }

    onMouseDown(event) {
    event.preventDefault();
    
    if (event.button === 0) {
        this.rotateStart.set(event.clientX, event.clientY);
        document.addEventListener('mousemove', this.onMouseMove);
        document.addEventListener('mouseup', this.onMouseUp);
    }
    }

    onMouseMove(event) {
    event.preventDefault();
    
    this.rotateEnd.set(event.clientX, event.clientY);
    this.rotateDelta.subVectors(this.rotateEnd, this.rotateStart).multiplyScalar(this.rotateSpeed);

    const element = this.domElement;
    this.sphericalDelta.theta -= 2 * Math.PI * this.rotateDelta.x / element.clientHeight;
    this.sphericalDelta.phi -= 2 * Math.PI * this.rotateDelta.y / element.clientHeight;

    this.rotateStart.copy(this.rotateEnd);
    this.update();
    }

    onMouseUp() {
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('mouseup', this.onMouseUp);
    }

    onMouseWheel(event) {
    event.preventDefault();
    
    if (event.deltaY < 0) {
        this.scale /= Math.pow(0.95, this.zoomSpeed);
    } else if (event.deltaY > 0) {
        this.scale *= Math.pow(0.95, this.zoomSpeed);
    }

    this.zoomChanged = true;
    this.update();
    }

    update() {
    const offset = new THREE.Vector3();
    const quat = new THREE.Quaternion().setFromUnitVectors(this.camera.up, new THREE.Vector3(0, 1, 0));
    const quatInverse = quat.clone().invert();

    const lastPosition = new THREE.Vector3();
    const lastQuaternion = new THREE.Quaternion();

    const position = this.camera.position;

    offset.copy(position).sub(this.target);
    offset.applyQuaternion(quat);

    this.spherical.setFromVector3(offset);

    if (this.autoRotate) {
        this.sphericalDelta.theta += 2 * Math.PI / 60 / 60 * this.autoRotateSpeed;
    }

    this.spherical.theta += this.sphericalDelta.theta;
    this.spherical.phi += this.sphericalDelta.phi;

    this.spherical.phi = Math.max(0.000001, Math.min(Math.PI - 0.000001, this.spherical.phi));
    this.spherical.radius *= this.scale;
    this.spherical.radius = Math.max(1, Math.min(100, this.spherical.radius));

    this.target.add(this.panOffset);

    offset.setFromSpherical(this.spherical);
    offset.applyQuaternion(quatInverse);

    position.copy(this.target).add(offset);
    this.camera.lookAt(this.target);

    this.sphericalDelta.set(0, 0, 0);
    this.scale = 1;
    this.panOffset.set(0, 0, 0);

    if (this.zoomChanged ||
        lastPosition.distanceToSquared(this.camera.position) > 0.000001 ||
        8 * (1 - lastQuaternion.dot(this.camera.quaternion)) > 0.000001) {
        
        lastPosition.copy(this.camera.position);
        lastQuaternion.copy(this.camera.quaternion);
        this.zoomChanged = false;
        
        return true;
    }

    return false;
    }

    dispose() {
    this.domElement.removeEventListener('mousedown', this.onMouseDown);
    this.domElement.removeEventListener('wheel', this.onMouseWheel);
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('mouseup', this.onMouseUp);
    }
}

export default OrbitControls;