// engine.js
import * as THREE from "three";
import { Roll } from "./roll";
import { createMaterials } from "./materials";
import { getFaceColorFactory } from "./colors";
import { moveMapping, rotateConditions } from "./constants";
import OrbitControls from "../OrbitControls";

export function createRubiksEngine({ mountEl, cubeData }) {
    const steps = 50;
    const stepAngle = Math.PI / (2 * steps);

    // scène + renderer + camera
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setClearColor("#000");
    renderer.setSize(mountEl.clientWidth, mountEl.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    const camera = new THREE.PerspectiveCamera(45,
        mountEl.clientWidth / mountEl.clientHeight, 1, 1000);
    camera.position.set(6, 6, 6);

    const controls = new OrbitControls(camera, renderer.domElement);
    mountEl.appendChild(renderer.domElement);

    // resize (ResizeObserver pour éviter les leaks)
    const ro = new ResizeObserver(([entry]) => {
        const { inlineSize: w, blockSize: h } = entry.contentBoxSize
            ? entry.contentBoxSize[0] : { inlineSize: mountEl.clientWidth, blockSize: mountEl.clientHeight };
        camera.aspect = w / h; camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    ro.observe(mountEl);

    // cubies
    const materials = createMaterials();
    const getFaceColor = getFaceColorFactory(cubeData);
    const cubes = [];
    const geom = new THREE.BoxGeometry(1, 1, 1);
    const coords = [-1, 0, 1];
    const createCubie = (x, y, z) => {
        const mats = Array.from({ length: 6 }, (_, f) => materials[getFaceColor(x, y, z, f)]);
        const mesh = new THREE.Mesh(geom, mats);
        mesh.position.set(x, y, z); cubes.push(mesh); scene.add(mesh);
    };
    coords.forEach(x => coords.forEach(y => coords.forEach(z => createCubie(x, y, z))));
    const pivot = new THREE.Group(); scene.add(pivot);

    // === Labels des faces (U,R,F,D,L,B) =========================
    const labelsGroup = new THREE.Group();
    labelsGroup.visible = false; 
    scene.add(labelsGroup);
    
 const _makeTextSprite = (text) => {
        const size = 256;
        const canvas = document.createElement("canvas");
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, size, size);
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "bold 160px sans-serif";
        ctx.fillText(text, size / 2, size / 2);
        const tex = new THREE.CanvasTexture(canvas);
        tex.needsUpdate = true;
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
        const sprite = new THREE.Sprite(mat);
        const scale = 0.9;
        sprite.scale.set(scale, scale, 1);
        sprite.userData.__labelTexture = tex;
        return sprite;
    };

    const _buildFaceLabels = () => {
        labelsGroup.clear();
        const d = 1.8; 
        const entries = [
            { txt: "R", pos: [d, 0, 0] }, // right
            { txt: "L", pos: [-d, 0, 0] }, // left
            { txt: "U", pos: [0, d, 0] }, // up
            { txt: "D", pos: [0, -d, 0] }, // down
            { txt: "F", pos: [0, 0, d] }, // front
            { txt: "B", pos: [0, 0, -d] }, // back
        ];
        entries.forEach(({ txt, pos }) => {
            const s = _makeTextSprite(txt);
            s.position.set(pos[0], pos[1], pos[2]);
            labelsGroup.add(s);
        });
    };
    _buildFaceLabels();


    // queue
    let roll = null;
    const queue = [];
    const startRoll = (faceName, dir) => {
        const cond = rotateConditions[faceName]; if (!cond) return;
        roll = new Roll({ faceCond: cond, direction: dir, scene, pivot, cubes, steps, stepAngle });
    };
    const enqueue = (move) => {
        if (!move) return;
        const k = String(move).trim().toUpperCase();
        const info = moveMapping[k]; if (!info) return;
        queue.push({ face: info.face, dir: info.dir });
        if (info.double) queue.push({ face: info.face, dir: info.dir });
    };
    const pump = () => {
        if (!roll && queue.length) { const n = queue.shift(); startRoll(n.face, n.dir); }
    };

    // boucle
    let raf = 0, running = true;
    const update = () => {
        if (roll) {
            if (roll.active) roll.rollFace();
            else { roll = null; pump(); }
        }
    };
    const loop = () => {
        if (!running) return;
        raf = requestAnimationFrame(loop);
        update();
        controls.update();
        renderer.render(scene, camera);
    };
    loop();

    const api = {
        enqueue, pump,
        addCubeRotation: (m) => { enqueue(m); pump(); },
        clearQueue: () => { queue.length = 0; },
        pause: () => { running = false; if (raf) cancelAnimationFrame(raf); },
        resume: () => { if (!running) { running = true; loop(); } },
        setFaceLabelsVisible: (v) => { labelsGroup.visible = !!v; },
        dispose: () => {
            running = false; if (raf) cancelAnimationFrame(raf);
            ro.disconnect(); controls.dispose(); renderer.dispose();
            if (renderer.domElement && mountEl.contains(renderer.domElement)) mountEl.removeChild(renderer.domElement);
                labelsGroup.traverse((obj) => {
                    if (obj.isSprite && obj.userData.__labelTexture) {
                        obj.userData.__labelTexture.dispose();
                    }
                });
            scene.clear();
        }
    };

    return api;
}