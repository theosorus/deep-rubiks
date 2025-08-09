// src/components/Rubiks/RubiksCube.js
import React, {
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
} from "react";
import * as THREE from "three";
import "./RubiksCube.css";
import OrbitControls from "./OrbitControls";

/**
 * Notation -> face logique + direction
 * Convention:
 *  - U,R,F,L,B,D : couches externes
 *  - M,E,S : tranches centrales (x=0, y=0, z=0)
 *  - dir: -1 = 90° dans le sens horaire (vue face), 1 = anti-horaire
 */
const moveMapping = {
  // Faces externes
  U: { face: "top", dir: -1 },
  "U'": { face: "top", dir: 1 },
  U2: { face: "top", dir: -1, double: true },

  R: { face: "right", dir: -1 },
  "R'": { face: "right", dir: 1 },
  R2: { face: "right", dir: -1, double: true },

  F: { face: "front", dir: -1 },
  "F'": { face: "front", dir: 1 },
  F2: { face: "front", dir: -1, double: true },

  L: { face: "left", dir: -1 },
  "L'": { face: "left", dir: 1 },
  L2: { face: "left", dir: -1, double: true },

  B: { face: "back", dir: -1 },
  "B'": { face: "back", dir: 1 },
  B2: { face: "back", dir: -1, double: true },

  D: { face: "bottom", dir: -1 },
  "D'": { face: "bottom", dir: 1 },
  D2: { face: "bottom", dir: -1, double: true },

  // Slices
  M: { face: "m", dir: -1 },
  "M'": { face: "m", dir: 1 },
  M2: { face: "m", dir: -1, double: true },

  E: { face: "e", dir: -1 },
  "E'": { face: "e", dir: 1 },
  E2: { face: "e", dir: -1, double: true },

  S: { face: "s", dir: -1 },
  "S'": { face: "s", dir: 1 },
  S2: { face: "s", dir: -1, double: true },
};

/**
 * Faces -> axe + valeur de couche à sélectionner
 * value: 1 = côté positif, -1 = côté négatif, 0 = centre
 */
const rotateConditions = {
  right: { axis: "x", value: 1 },
  left: { axis: "x", value: -1 },
  top: { axis: "y", value: 1 },
  bottom: { axis: "y", value: -1 },
  front: { axis: "z", value: 1 },
  back: { axis: "z", value: -1 },

  // Slices
  m: { axis: "x", value: 0 },
  e: { axis: "y", value: 0 },
  s: { axis: "z", value: 0 },
};

const RubiksCube = forwardRef(({ cubeData }, ref) => {
  const mountRef = useRef(null);

  // Refs moteur
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const pivotRef = useRef(null);
  const cubesRef = useRef([]);
  const rollRef = useRef(null);
  const queueRef = useRef([]);
  const animationIdRef = useRef(null);

  // API impérative exposée au parent
  const apiRef = useRef({ enqueue: () => {}, pump: () => {} });

  // ----- util: mapping de couleurs à partir du payload du back -----
  const getFaceColorFactory = (data) => {
    // Laisse le back préciser l'ordre. Fallback configurable ici si manquant.
    // Ajuste ce DEFAULT_FACE_ORDER si besoin pour coller à ton cube.py
    const DEFAULT_FACE_ORDER = ["U","R","F","D","L","B"];
    const FACE_ORDER =
      Array.isArray(data?.faceOrder) && data.faceOrder.length === 6
        ? data.faceOrder
        : DEFAULT_FACE_ORDER;

    const idx = (name) => {
      const i = FACE_ORDER.indexOf(name);
      return i === -1 ? 0 : i;
    };

    return (x, y, z, faceIndex) => {
      const conv = (n) => n + 1; // [-1..1] -> [0..2]

      // faceIndex (0..5) = ordre THREE: right,left,top,bottom,front,back
      const faceMap = [
        { face: idx("R"), row: conv(y), col: 2 - conv(z) }, // right
        { face: idx("L"), row: conv(y), col: conv(z) }, // left
        { face: idx("U"), row: 2 - conv(z), col: conv(x) }, // top
        { face: idx("D"), row: conv(z), col: conv(x) }, // bottom
        { face: idx("F"), row: conv(y), col: conv(x) }, // front
        { face: idx("B"), row: conv(y), col: 2 - conv(x) }, // back
      ];

      const isVisible = (fi, xx, yy, zz) => {
        switch (fi) {
          case 0:
            return xx === 1;
          case 1:
            return xx === -1;
          case 2:
            return yy === 1;
          case 3:
            return yy === -1;
          case 4:
            return zz === 1;
          case 5:
            return zz === -1;
          default:
            return false;
        }
      };

      if (!isVisible(faceIndex, x, y, z)) return "gray";
      const m = faceMap[faceIndex];
      const colorNumber = data.state?.[m.face]?.[m.row]?.[m.col];
      const colorName = data.colors?.[String(colorNumber)];
      return colorName || "gray";
    };
  };

  useEffect(() => {
    if (!cubeData) return;

    // --- Constantes d’anim
    const steps = 50; // frames pour 90°
    const stepAngle = Math.PI / (2 * steps);

    // --- Shaders pour un fin liseré
    const vertexShader = `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
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
    const createMat = (hex) =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: { faceColor: { value: new THREE.Color(hex) } },
      });

    const materials = {
      blue: createMat("#0000FF"),
      red: createMat("#FF0000"),
      white: createMat("#FFFFFF"),
      green: createMat("#009B48"),
      yellow: createMat("#FFFF00"),
      orange: createMat("#FFA500"),
      gray: createMat("#777777"),
    };

    const getFaceColor = getFaceColorFactory(cubeData);

    // --- Classe: rotation d'une couche
    class Roll {
      constructor(faceCond, direction, scene, pivot, cubes) {
        this.face = faceCond; // {axis, value}
        this.dir = direction; // -1 | 1
        this.scene = scene;
        this.pivot = pivot;
        this.cubes = cubes;
        this.stepCount = 0;
        this.active = true;
        this._pickLayer();
      }

      _pickLayer() {
        this.cubes.forEach((m) => {
          if (m.position[this.face.axis] === this.face.value) {
            this.scene.remove(m);
            this.pivot.add(m);
          }
        });
      }

      rollFace() {
        if (this.stepCount < steps) {
          this.pivot.rotation[this.face.axis] += this.dir * stepAngle;
          this.stepCount++;
          return;
        }
        if (!this.active) return;
        this.active = false;
        this._bake();
      }

      _bake() {
        const q = new THREE.Quaternion();
        this.pivot.updateWorldMatrix(true, true);

        for (let i = this.pivot.children.length - 1; i >= 0; i--) {
          const m = this.pivot.children[i];
          m.updateWorldMatrix(true, false);

          // position/rotation monde -> locales pour la scène
          m.getWorldPosition(m.position);
          m.getWorldQuaternion(q);
          m.quaternion.copy(q);

          // arrondi des coords pour éviter les flottants
          m.position.x = Math.round(m.position.x);
          m.position.y = Math.round(m.position.y);
          m.position.z = Math.round(m.position.z);

          this.pivot.remove(m);
          this.scene.add(m);
        }
        this.pivot.rotation.set(0, 0, 0);
      }
    }

    // --- Init scène
    const root = mountRef.current;
    const { clientWidth, clientHeight } = root;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor("#000");
    renderer.setSize(clientWidth, clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    const camera = new THREE.PerspectiveCamera(
      45,
      clientWidth / clientHeight,
      1,
      1000
    );
    camera.position.set(6, 6, 6);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;

    root.appendChild(renderer.domElement);

    const onWindowResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onWindowResize, false);

    // --- Création des 27 cubies
    const cubes = [];
    const geom = new THREE.BoxGeometry(1, 1, 1);
    const coords = [-1, 0, 1];

    const createCubie = (x, y, z) => {
      const mats = [];
      for (let f = 0; f < 6; f++) {
        mats.push(materials[getFaceColor(x, y, z, f)]);
      }
      const mesh = new THREE.Mesh(geom, mats);
      mesh.position.set(x, y, z);
      cubes.push(mesh);
      scene.add(mesh);
    };

    coords.forEach((x) =>
      coords.forEach((y) => coords.forEach((z) => createCubie(x, y, z)))
    );

    const pivot = new THREE.Group();
    scene.add(pivot);

    cubesRef.current = cubes;
    pivotRef.current = pivot;

    // --- Queue & boucle d’anim
    const startRoll = (faceName, dir) => {
      const cond = rotateConditions[faceName];
      if (!cond) return;
      rollRef.current = new Roll(cond, dir, scene, pivot, cubes);
    };

    const enqueueMove = (move) => {
      if (!move) return;
      const k = String(move).trim().toUpperCase(); // case-insensitive
      const info = moveMapping[k];
      if (!info) return;
      queueRef.current.push({ face: info.face, dir: info.dir });
      if (info.double) queueRef.current.push({ face: info.face, dir: info.dir });
    };

    const pumpQueueIfIdle = () => {
      if (!rollRef.current && queueRef.current.length > 0) {
        const next = queueRef.current.shift();
        startRoll(next.face, next.dir);
      }
    };

    // Expo API
    apiRef.current.enqueue = enqueueMove;
    apiRef.current.pump = pumpQueueIfIdle;

    const update = () => {
      const r = rollRef.current;
      if (r) {
        if (r.active) r.rollFace();
        else {
          rollRef.current = null;
          pumpQueueIfIdle();
        }
      }
    };

    const loop = () => {
      animationIdRef.current = requestAnimationFrame(loop);
      update();
      controls.update();
      renderer.render(scene, camera);
    };
    loop();

    // --- Cleanup
    return () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      window.removeEventListener("resize", onWindowResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement && root?.contains(renderer.domElement)) {
        root.removeChild(renderer.domElement);
      }
      scene.clear();
    };
  }, [cubeData]);

  // ----- API impérative pour App.js : clic bouton -> anim immédiate -----
  useImperativeHandle(ref, () => ({
    addCubeRotation: (move) => {
      apiRef.current.enqueue(move);
      apiRef.current.pump();
    },
  }));

  if (!cubeData) {
    return <div id="loading-message">Chargement du cube…</div>;
  }

  return (
    <div
      ref={mountRef}
      style={{
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: "#000",
      }}
    />
  );
});

export default RubiksCube;