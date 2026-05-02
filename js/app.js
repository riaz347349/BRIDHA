/**
 * Bridha AR Dress Visualiser
 * Uses TensorFlow.js MoveNet for real-time pose detection and
 * overlays bridal dress silhouettes onto the live camera feed.
 */

/* ── Keypoint indices (MoveNet / COCO) ──────────────────────────── */
const KP = {
    NOSE: 0,
    LEFT_SHOULDER: 5, RIGHT_SHOULDER: 6,
    LEFT_ELBOW: 7,    RIGHT_ELBOW: 8,
    LEFT_WRIST: 9,    RIGHT_WRIST: 10,
    LEFT_HIP: 11,     RIGHT_HIP: 12,
    LEFT_KNEE: 13,    RIGHT_KNEE: 14,
    LEFT_ANKLE: 15,   RIGHT_ANKLE: 16,
};

/* ── Exponential smoothing factor (0 = frozen, 1 = raw) ─────────── */
const ALPHA = 0.35;

/* ── SVG dress silhouettes keyed by style ───────────────────────── */
function dresseSVG(dress) {
    const c  = dress.color || '#fff8f0';
    const s  = shadeHex(c, -18);
    const ac = '#d0c0a8';
    const id = dress.id;

    const grad = (id) => `
        <defs>
            <linearGradient id="dg${id}" x1="0%" y1="0%" x2="60%" y2="100%">
                <stop offset="0%"   stop-color="${c}"/>
                <stop offset="100%" stop-color="${s}"/>
            </linearGradient>
        </defs>`;

    const shapes = {

        aline: `<svg viewBox="0 0 120 210" xmlns="http://www.w3.org/2000/svg">
            ${grad(id)}
            <!-- bodice -->
            <path d="M34 6 Q28 2 35 0 L85 0 Q92 2 86 6 L88 82 Q60 92 32 82Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- strapless sweetheart -->
            <path d="M34 6 Q47 20 60 14 Q73 20 86 6" fill="url(#dg${id})" stroke="${ac}" stroke-width="1.2"/>
            <!-- skirt A-line -->
            <path d="M32 82 Q8 130 0 210 L120 210 Q112 130 88 82Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- lace rows bodice -->
            <path d="M35 22 Q60 32 85 22" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M34 44 Q60 54 86 44" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M33 64 Q60 74 87 64" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <!-- skirt flow lines -->
            <path d="M58 95 Q50 155 44 210" fill="none" stroke="${ac}" stroke-width="0.5" opacity="0.45"/>
            <path d="M62 95 Q70 155 76 210" fill="none" stroke="${ac}" stroke-width="0.5" opacity="0.45"/>
        </svg>`,

        ballgown: `<svg viewBox="0 0 130 210" xmlns="http://www.w3.org/2000/svg">
            ${grad(id)}
            <!-- bodice -->
            <path d="M36 6 Q30 0 38 0 L92 0 Q100 0 94 6 L92 82 Q65 94 38 82Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- sweetheart -->
            <path d="M36 6 Q50 22 65 15 Q80 22 94 6" fill="url(#dg${id})" stroke="${ac}" stroke-width="1.2"/>
            <!-- full ballgown skirt -->
            <path d="M38 82 Q-8 105 -12 210 L142 210 Q138 105 92 82Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- tulle waves -->
            <path d="M14 128 Q28 120 42 130 Q56 120 70 130 Q84 120 98 128 Q110 122 118 128"
                  fill="none" stroke="${ac}" stroke-width="0.9" opacity="0.55"/>
            <path d="M4 158 Q22 148 40 158 Q58 148 76 158 Q94 148 112 155 Q122 150 126 156"
                  fill="none" stroke="${ac}" stroke-width="0.9" opacity="0.45"/>
            <path d="M0 185 Q22 174 44 185 Q66 174 88 185 Q108 176 124 183"
                  fill="none" stroke="${ac}" stroke-width="0.9" opacity="0.35"/>
            <!-- bodice lace -->
            <path d="M39 22 Q65 32 91 22" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M38 44 Q65 54 92 44" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M37 66 Q65 76 93 66" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
        </svg>`,

        mermaid: `<svg viewBox="0 0 110 220" xmlns="http://www.w3.org/2000/svg">
            ${grad(id)}
            <!-- fitted body + mermaid flare -->
            <path d="M36 0 L74 0 L78 80 Q82 130 76 148 Q94 162 98 210 L12 210 Q16 162 34 148 Q28 130 32 80Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- flare -->
            <path d="M12 210 Q4 192 34 160 Q55 172 76 160 Q106 192 98 210Z"
                  fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- neckline -->
            <path d="M36 0 Q55 14 74 0" fill="none" stroke="${ac}" stroke-width="1.2"/>
            <!-- fitted seam lines + lace rows -->
            <path d="M36 20 Q55 30 74 20" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M35 45 Q55 55 75 45" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M34 72 Q55 82 76 72" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M33 100 Q55 110 77 100" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <line x1="55" y1="0" x2="55" y2="148" stroke="${ac}" stroke-width="0.4" opacity="0.3"/>
        </svg>`,

        sheath: `<svg viewBox="0 0 110 210" xmlns="http://www.w3.org/2000/svg">
            ${grad(id)}
            <!-- slim sheath -->
            <path d="M36 0 L74 0 L78 205 L32 205Z" fill="url(#dg${id})" stroke="${ac}" stroke-width="0.6"/>
            <!-- neckline illusion -->
            <path d="M36 0 Q55 16 74 0" fill="none" stroke="${ac}" stroke-width="1.2"/>
            <path d="M36 0 Q55 20 74 0" fill="none" stroke="${ac}" stroke-width="0.5" stroke-dasharray="1.5 2.5" opacity="0.5"/>
            <!-- horizontal lace rows -->
            <path d="M37 28 Q55 38 73 28" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M36 62 Q55 72 74 62" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M35 100 Q55 110 75 100" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M35 140 Q55 150 75 140" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <path d="M34 180 Q55 190 76 180" fill="none" stroke="${ac}" stroke-width="0.7" stroke-dasharray="2 3"/>
            <!-- side seam detail -->
            <line x1="48" y1="0" x2="50" y2="205" stroke="${ac}" stroke-width="0.4" opacity="0.25"/>
            <line x1="62" y1="0" x2="60" y2="205" stroke="${ac}" stroke-width="0.4" opacity="0.25"/>
        </svg>`,
    };

    return shapes[dress.style] || shapes.aline;
}

/** Darken a hex colour by `amount` (0–255) */
function shadeHex(hex, amount) {
    const clean = hex.replace('#', '');
    const full  = clean.length === 3
        ? clean.split('').map(x => x + x).join('')
        : clean;
    const num = parseInt(full, 16);
    const clamp = v => Math.max(0, Math.min(255, v));
    const r = clamp((num >> 16) + amount);
    const g = clamp(((num >> 8) & 0xff) + amount);
    const b = clamp((num & 0xff) + amount);
    return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

/* ── Main Application ───────────────────────────────────────────── */
class DressVisualiser {
    constructor() {
        this.video      = document.getElementById('video');
        this.canvas     = document.getElementById('canvas');
        this.ctx        = this.canvas.getContext('2d');
        this.detector   = null;
        this.selectedDress    = null;
        this.dressImageCache  = {};
        this.smoothed         = null;
        this.filter           = 'all';
        this.opacity          = 0.88;
        this.running          = false;
    }

    async init() {
        this.setLoadingText('Requesting camera…');

        try {
            await this.startCamera();
        } catch (err) {
            console.error('Camera error:', err);
            document.getElementById('loading-screen').classList.add('hidden');
            document.getElementById('error-screen').classList.remove('hidden');
            return;
        }

        this.setLoadingText('Loading AI model… (first load may take ~15s)');

        // Warn user if taking a long time
        const slowTimer = setTimeout(() => {
            this.setLoadingText('Still loading — please keep the page open…');
        }, 12000);

        try {
            await this.loadPoseModel();
        } catch (err) {
            clearTimeout(slowTimer);
            console.error('Model load error:', err);
            this.showErrorScreen('Failed to load AI model. Please check your internet connection and reload.');
            return;
        }

        clearTimeout(slowTimer);
        this.setLoadingText('Preparing dresses…');
        this.buildUI();
        this.selectDress(DRESSES[0]);

        // Fade out loader
        const loader = document.getElementById('loading-screen');
        loader.style.opacity = '0';
        loader.style.transition = 'opacity 0.5s';
        setTimeout(() => {
            loader.classList.add('hidden');
            document.getElementById('ar-view').classList.remove('hidden');
            this.running = true;
            this.loop();
        }, 500);
    }

    setLoadingText(t) {
        document.getElementById('loading-text').textContent = t;
    }

    showErrorScreen(msg) {
        document.getElementById('loading-screen').classList.add('hidden');
        const err = document.getElementById('error-screen');
        const p = err.querySelector('p');
        if (p && msg) p.textContent = msg;
        err.classList.remove('hidden');
    }

    /* ── Camera ─────────────────────────────────────────────────── */
    async startCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width:  { ideal: 720 },
                height: { ideal: 1280 },
            },
            audio: false,
        });

        this.video.srcObject = stream;

        await new Promise((res, rej) => {
            this.video.onloadedmetadata = res;
            this.video.onerror = rej;
        });

        await this.video.play();
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const vw = this.video.videoWidth  || window.innerWidth;
        const vh = this.video.videoHeight || window.innerHeight;
        this.canvas.width  = vw;
        this.canvas.height = vh;
    }

    /* ── Pose model ─────────────────────────────────────────────── */
    async loadPoseModel() {
        // Try backends in order of preference — WebGL is fastest but WASM
        // is a reliable fallback on older Android devices
        const backends = ['webgl', 'wasm', 'cpu'];
        let backendReady = false;

        for (const backend of backends) {
            try {
                await tf.setBackend(backend);
                await tf.ready();
                console.log('TF backend:', backend);
                backendReady = true;
                break;
            } catch (e) {
                console.warn(`Backend "${backend}" unavailable:`, e.message);
            }
        }

        if (!backendReady) throw new Error('No TF backend available');

        this.detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            {
                modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
                enableSmoothing: true,
            }
        );
    }

    /* ── UI setup ───────────────────────────────────────────────── */
    buildUI() {
        // Opacity slider
        const slider = document.getElementById('opacity-slider');
        slider.addEventListener('input', () => {
            this.opacity = parseInt(slider.value) / 100;
        });

        // Capture button
        document.getElementById('capture-btn').addEventListener('click', () => this.capture());

        // Designer tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.filter = tab.dataset.designer;
                this.renderCarousel();
            });
        });

        this.renderCarousel();
    }

    renderCarousel() {
        const carousel = document.getElementById('dress-carousel');
        const list = this.filter === 'all'
            ? DRESSES
            : DRESSES.filter(d => d.designer === this.filter);

        carousel.innerHTML = '';

        list.forEach(dress => {
            const card = document.createElement('div');
            card.className = 'dress-card' + (this.selectedDress?.id === dress.id ? ' selected' : '');
            card.setAttribute('role', 'option');
            card.setAttribute('aria-selected', this.selectedDress?.id === dress.id);
            card.dataset.id = dress.id;

            card.innerHTML = `
                <div class="dress-thumb">${dresseSVG(dress)}</div>
                <div class="dress-card-name">${dress.name}</div>
                <div class="dress-card-designer">${dress.designerLabel}</div>`;

            card.addEventListener('click', () => this.selectDress(dress));
            carousel.appendChild(card);
        });
    }

    selectDress(dress) {
        this.selectedDress = dress;
        document.getElementById('dress-name').textContent     = dress.name;
        document.getElementById('dress-designer').textContent = dress.designerLabel;

        document.querySelectorAll('.dress-card').forEach(c => {
            const sel = c.dataset.id === dress.id;
            c.classList.toggle('selected', sel);
            c.setAttribute('aria-selected', sel);
        });

        // Pre-render dress to an offscreen Image for fast canvas drawing
        this.currentDressImg = null;
        this.svgToImage(dress).then(img => { this.currentDressImg = img; });
    }

    svgToImage(dress) {
        // Return from cache if available
        if (this.dressImageCache[dress.id]) {
            return Promise.resolve(this.dressImageCache[dress.id]);
        }

        // If a real photo URL is set, load that
        if (dress.imageUrl) {
            return new Promise(res => {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => { this.dressImageCache[dress.id] = img; res(img); };
                img.onerror = () => res(this.svgFallback(dress));
                img.src = dress.imageUrl;
            });
        }

        return this.svgFallback(dress);
    }

    svgFallback(dress) {
        return new Promise(res => {
            const svg  = dresseSVG(dress);
            const blob = new Blob([svg], { type: 'image/svg+xml' });
            const url  = URL.createObjectURL(blob);
            const img  = new Image();
            img.onload = () => {
                URL.revokeObjectURL(url);
                this.dressImageCache[dress.id] = img;
                res(img);
            };
            img.src = url;
        });
    }

    /* ── Render loop ─────────────────────────────────────────────── */
    loop() {
        if (!this.running) return;
        this.frame().finally(() => requestAnimationFrame(() => this.loop()));
    }

    async frame() {
        const { canvas, ctx, video } = this;
        const vw = canvas.width;
        const vh = canvas.height;

        if (!vw || !vh || video.readyState < 2) return;

        // Draw mirrored video
        ctx.save();
        ctx.translate(vw, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, vw, vh);
        ctx.restore();

        // Detect pose
        let poses = [];
        try { poses = await this.detector.estimatePoses(video); } catch { return; }

        if (poses.length > 0) {
            this.updateSmoothed(poses[0].keypoints, vw);
        }

        if (this.poseReady()) {
            document.getElementById('guide-overlay').classList.add('hidden');
            this.drawDress(vw, vh);
        } else {
            document.getElementById('guide-overlay').classList.remove('hidden');
            document.getElementById('guide-text').textContent =
                poses.length === 0
                    ? 'Stand in front of the camera'
                    : 'Step back so your full body is visible';
        }
    }

    /* ── Smoothing ───────────────────────────────────────────────── */
    updateSmoothed(keypoints, vw) {
        // Mirror X so keypoints align with the mirrored video frame
        const mirrored = keypoints.map(kp => ({ ...kp, x: vw - kp.x }));

        if (!this.smoothed) {
            this.smoothed = mirrored.map(kp => ({ ...kp }));
            return;
        }

        this.smoothed = this.smoothed.map((sk, i) => ({
            ...sk,
            x:     sk.x     * (1 - ALPHA) + mirrored[i].x     * ALPHA,
            y:     sk.y     * (1 - ALPHA) + mirrored[i].y     * ALPHA,
            score: mirrored[i].score,
        }));
    }

    poseReady() {
        if (!this.smoothed) return false;
        const needed = [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP];
        return needed.every(i => (this.smoothed[i]?.score ?? 0) > 0.25);
    }

    /* ── Dress overlay ───────────────────────────────────────────── */
    drawDress(vw, vh) {
        if (!this.currentDressImg || !this.smoothed) return;

        const kp  = this.smoothed;
        const ls  = kp[KP.LEFT_SHOULDER];
        const rs  = kp[KP.RIGHT_SHOULDER];
        const lh  = kp[KP.LEFT_HIP];
        const rh  = kp[KP.RIGHT_HIP];
        const lk  = kp[KP.LEFT_KNEE];
        const rk  = kp[KP.RIGHT_KNEE];
        const la  = kp[KP.LEFT_ANKLE];
        const ra  = kp[KP.RIGHT_ANKLE];

        const shoulderCX = (ls.x + rs.x) / 2;
        const shoulderCY = (ls.y + rs.y) / 2;
        const shoulderW  = Math.abs(rs.x - ls.x);

        // Dress top: just above shoulders to cover décolletage
        const dressTopY  = shoulderCY - shoulderW * 0.28;

        // Dress bottom: prefer ankles → knees → estimate
        let dressBottomY;
        const ankleConf = Math.max(la?.score ?? 0, ra?.score ?? 0);
        const kneeConf  = Math.max(lk?.score ?? 0, rk?.score ?? 0);

        if (ankleConf > 0.2) {
            const ankleY = Math.max(
                (la?.score ?? 0) > 0.2 ? la.y : 0,
                (ra?.score ?? 0) > 0.2 ? ra.y : 0,
            );
            dressBottomY = ankleY + shoulderW * 0.25;
        } else if (kneeConf > 0.2) {
            const kneeY = Math.max(
                (lk?.score ?? 0) > 0.2 ? lk.y : 0,
                (rk?.score ?? 0) > 0.2 ? rk.y : 0,
            );
            const torso  = kneeY - shoulderCY;
            dressBottomY = shoulderCY + torso * 2.15;
        } else {
            const hipCY  = ((lh.y + rh.y) / 2);
            const torso  = hipCY - shoulderCY;
            dressBottomY = shoulderCY + torso * 3.6;
        }

        // Clamp to canvas
        dressBottomY = Math.min(dressBottomY, vh + shoulderW * 0.3);

        const dressH = dressBottomY - dressTopY;
        const dressW = shoulderW * 2.85;
        const dressX = shoulderCX - dressW / 2;

        this.ctx.save();
        this.ctx.globalAlpha = this.opacity;
        this.ctx.drawImage(this.currentDressImg, dressX, dressTopY, dressW, dressH);
        this.ctx.restore();
    }

    /* ── Screenshot ──────────────────────────────────────────────── */
    capture() {
        // Flash effect
        const flash = document.createElement('div');
        flash.className = 'flash';
        document.body.appendChild(flash);
        setTimeout(() => flash.remove(), 500);

        const name = this.selectedDress?.name?.toLowerCase().replace(/\s+/g, '-') || 'look';
        const link = document.createElement('a');
        link.download = `bridha-${name}.png`;
        link.href     = this.canvas.toDataURL('image/png');
        link.click();
    }
}

/* ── Bootstrap ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    new DressVisualiser().init();
});
