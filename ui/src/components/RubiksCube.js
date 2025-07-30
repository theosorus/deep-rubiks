import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const RubiksCube = ({ cubeData }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const animationIdRef = useRef(null);

  useEffect(() => {
    if (!cubeData) return; 

    let scene, camera, renderer, controls, rollObject, group;
    let cubes = [];

    const rotateConditions = {
      right: { axis: "x", value: 1 },
      left: { axis: "x", value: -1 },
      top: { axis: "y", value: 1 },
      bottom: { axis: "y", value: -1 },
      front: { axis: "z", value: 1 },
      back: { axis: "z", value: -1 }
    };

    // Fonction pour obtenir la couleur d'une face spécifique d'un cube
    const getFaceColor = (x, y, z, faceIndex) => {
      // Conversion des coordonnées Three.js (-1,0,1) vers indices de tableau (0,1,2)
      const convertCoord = (coord) => coord + 1;
      
      // Mappage des faces Three.js vers les faces de l'API
      // faceIndex correspond à l'ordre des matériaux dans Three.js: [right, left, top, bottom, front, back]
      const faceMapping = [
        { face: 0, row: convertCoord(y), col: 2 - convertCoord(z) }, // right (x=1) -> face 0
        { face: 1, row: convertCoord(y), col: convertCoord(z) },     // left (x=-1) -> face 1  
        { face: 2, row: 2 - convertCoord(z), col: convertCoord(x) }, // top (y=1) -> face 2
        { face: 3, row: convertCoord(z), col: convertCoord(x) },     // bottom (y=-1) -> face 3
        { face: 4, row: convertCoord(y), col: convertCoord(x) },     // front (z=1) -> face 4
        { face: 5, row: convertCoord(y), col: 2 - convertCoord(x) }  // back (z=-1) -> face 5
      ];

      // Vérifier si cette face est visible (sur le bord du cube)
      const isVisibleFace = (faceIdx, x, y, z) => {
        switch(faceIdx) {
          case 0: return x === 1;  // right
          case 1: return x === -1; // left
          case 2: return y === 1;  // top
          case 3: return y === -1; // bottom
          case 4: return z === 1;  // front
          case 5: return z === -1; // back
          default: return false;
        }
      };

      if (!isVisibleFace(faceIndex, x, y, z)) {
        return "gray"; // Face interne
      }

      const mapping = faceMapping[faceIndex];
      const colorNumber = cubeData.state[mapping.face][mapping.row][mapping.col];
      const colorName = cubeData.colors[colorNumber.toString()];
      
      return colorName || "gray";
    };

    const step = Math.PI / 100;
    const faces = ["front", "back", "left", "right", "top", "bottom"];
    const directions = [-1, 1];
    const cPositions = [-1, 0, 1];

    const vertexShader = `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
      }
    `;

    const fragmentShader = `
      varying vec2 vUv;
      uniform vec3 faceColor;
      void main() {
          vec3 border = vec3(0.533);
          float bl = smoothstep(0.0, 0.03, vUv.x);
          float br = smoothstep(1.0, 0.97, vUv.x);
          float bt = smoothstep(0.0, 0.03, vUv.y);
          float bb = smoothstep(1.0, 0.97, vUv.y);
          vec3 c = mix(border, faceColor, bt*br*bb*bl);
          gl_FragColor = vec4(c, 1.0);
      }
    `;

    const createMaterial = (color) =>
      new THREE.ShaderMaterial({
        fragmentShader,
        vertexShader,
        uniforms: { faceColor: { type: "v3", value: color } }
      });

    const materials = Object.entries({
      blue: new THREE.Color("#0000FF"),
      red: new THREE.Color("#FF0000"),
      white: new THREE.Color("#FFFFFF"),
      green: new THREE.Color("#009B48"),
      yellow: new THREE.Color("#FFFF00"),
      orange: new THREE.Color("#FFA500"),
      gray: new THREE.Color("#000000ff"),
    }).reduce((acc, [key, val]) => ({ ...acc, [key]: createMaterial(val) }), {});

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

    class Roll {
      constructor(face, direction) {
        this.face = face;
        this.stepCount = 0;
        this.active = true;
        this.direction = direction;
        this.init();
      }

      init() {
        cubes.forEach((item) => {
          if (item.position[this.face.axis] == this.face.value) {
            scene.remove(item);
            group.add(item);
          }
        });
      }

      rollFace() {
        if (this.stepCount != 50) {
          group.rotation[this.face.axis] += this.direction * step;
          this.stepCount += 1;
        } else {
          if (this.active) {
            this.active = false;
            this.clearGroup();
          }
        }
      }

      clearGroup() {
        for (var i = group.children.length - 1; i >= 0; i--) {
          let item = group.children[i];
          item.getWorldPosition(item.position);
          item.getWorldQuaternion(item.rotation);
          item.position.x = Math.round(item.position.x);
          item.position.y = Math.round(item.position.y);
          item.position.z = Math.round(item.position.z);
          group.remove(item);
          scene.add(item);
        }
        group.rotation[this.face.axis] = 0;
      }
    }

    function init() {
      const { clientHeight, clientWidth } = mountRef.current;
      scene = new THREE.Scene();
      sceneRef.current = scene;

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setClearColor("#000");
      renderer.setSize(clientWidth, clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      
      camera = new THREE.PerspectiveCamera(45, clientWidth / clientHeight, 1, 1000);
      camera.position.set(6, 6, 6);
      
      controls = new OrbitControls(camera, renderer.domElement);

      mountRef.current.appendChild(renderer.domElement);

      window.addEventListener("resize", onWindowResize, false);
      createObjects();
    }

    function onWindowResize() {
      if (!mountRef.current) return;
      const { clientWidth, clientHeight } = mountRef.current;
      camera.aspect = clientWidth / clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(clientWidth, clientHeight);
    }

    function createObjects() {
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      
      let createCube = (position) => {
        let mat = [];
        // Créer les matériaux pour chaque face dans l'ordre: [right, left, top, bottom, front, back]
        for (let i = 0; i < 6; i++) {
          const colorName = getFaceColor(position.x, position.y, position.z, i);
          mat.push(materials[colorName]);
        }
        
        const cube = new THREE.Mesh(geometry, mat);
        cube.position.set(position.x, position.y, position.z);
        cubes.push(cube);
        scene.add(cube);
      };

      cPositions.forEach((x) => {
        cPositions.forEach((y) => {
          cPositions.forEach((z) => {
            createCube({ x, y, z });
          });
        });
      });

      group = new THREE.Group();
      scene.add(group);
      rollObject = new Roll(rotateConditions["top"], -1);
    }

    function update() {
      if (rollObject) {
        if (rollObject.active) {
          rollObject.rollFace();
        } else {
          rollObject = new Roll(
            rotateConditions[faces[Math.floor(Math.random() * faces.length)]],
            directions[Math.floor(Math.random() * directions.length)]
          );
        }
      }
    }

    function render() {
      if (!mountRef.current) return;
      
      animationIdRef.current = requestAnimationFrame(render);
      update();
      controls.update();
      renderer.render(scene, camera);
    }

    init();
    render();

    // Cleanup function
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      
      if (controls) {
        controls.dispose();
      }
      
      if (renderer) {
        renderer.dispose();
      }
      
      if (mountRef.current && renderer) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      window.removeEventListener("resize", onWindowResize);
    };
  }, [cubeData]); // Dépendance sur cubeData

  if (!cubeData) {
    return (
      <div style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#000',
        color: '#fff',
        fontSize: '24px'
      }}>
        Chargement du cube...
      </div>
    );
  }

  return (
    <div 
      ref={mountRef} 
      style={{ 
        width: '100vw', 
        height: '100vh', 
        overflow: 'hidden',
        background: '#000'
      }} 
    />
  );
};

export default RubiksCube;