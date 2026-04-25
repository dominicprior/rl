import './style.css'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
    <h2>Cliff Walking Reinforcement Learning</h2>
    <div class="controls">
        Algorithm: 
        <select id="algoSelect">
            <option value="qlearning">Q-Learning (Off-policy)</option>
            <option value="sarsa">SARSA (On-policy)</option>
            <option value="montecarlo">Monte Carlo (Every-visit)</option>
        </select>
        <button onclick="resetSim()">Reset & Run</button>
        <div class="stats" id="stats">Episode: 0 | Steps: 0 | Reward: 0</div>
        <small>Blue = Agent | Red = Cliff | Green = Goal | Yellow arrows = Policy</small>
    </div>
    <canvas id="gridCanvas" width="600" height="240"></canvas>
`
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');
const ROWS = 4, COLS = 10, TILE = 60;
const ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

let state = { x: 0, y: 3 };
let qTable = {};
let episode = 0, steps = 0, totalReward = 0;
let running = true;
let history = []; // For Monte Carlo

function getQ(s, a) {
    if (!qTable[s]) qTable[s] = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
    return qTable[s][a];
}

function chooseAction(s, epsilon = 0.1) {
    if (Math.random() < epsilon) return ACTIONS[Math.floor(Math.random() * 4)];
    let scores = ACTIONS.map(a => getQ(s, a));
    return ACTIONS[scores.indexOf(Math.max(...scores))];
}

function move(s, a) {
    let next = { x: s.x, y: s.y };
    if (a === 'UP') next.y = Math.max(0, s.y - 1);
    if (a === 'DOWN') next.y = Math.min(ROWS - 1, s.y + 1);
    if (a === 'LEFT') next.x = Math.max(0, s.x - 1);
    if (a === 'RIGHT') next.x = Math.min(COLS - 1, s.x + 1);
    
    let reward = -1;
    let done = false;

    // The Cliff
    if (next.y === 3 && next.x > 0 && next.x < COLS - 1) {
        reward = -100;
        next = { x: 0, y: 3 }; // Back to start
    } else if (next.x === COLS - 1 && next.y === 3) {
        reward = 0;
        done = true;
    }
    return { next, reward, done };
}

async function loop() {
    let algo = document.getElementById('algoSelect').value;
    let s = `${state.x},${state.y}`;
    let a = chooseAction(s);
    
    while (running) {
        let { next, reward, done } = move(state, a);
        let sNext = `${next.x},${next.y}`;
        let aNext = chooseAction(sNext);

        if (algo === 'qlearning') {
            let maxNextQ = Math.max(...ACTIONS.map(act => getQ(sNext, act)));
            qTable[s][a] += 0.1 * (reward + 0.9 * maxNextQ - qTable[s][a]);
        } else if (algo === 'sarsa') {
            qTable[s][a] += 0.1 * (reward + 0.9 * getQ(sNext, aNext) - qTable[s][a]);
        } else if (algo === 'montecarlo') {
            history.push({ s, a, r: reward });
        }

        state = next;
        s = sNext;
        a = aNext;
        steps++;
        totalReward += reward;

        draw();
        document.getElementById('stats').innerText = `Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;

        if (done || steps > 500) {
            if (algo === 'montecarlo') {
                let G = 0;
                for (let i = history.length - 1; i >= 0; i--) {
                    G = history[i].r + 0.9 * G;
                    qTable[history[i].s][history[i].a] += 0.05 * (G - qTable[history[i].s][history[i].a]);
                }
                history = [];
            }
            episode++;
            steps = 0;
            totalReward = 0;
            state = { x: 0, y: 3 };
            s = `0,3`;
            a = chooseAction(s);
        }
        await new Promise(r => setTimeout(r, 10));
    }
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            let s = `${x},${y}`;
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(x * TILE, y * TILE, TILE, TILE);
            
            // Draw Cliff
            if (y === 3 && x > 0 && x < COLS - 1) {
                ctx.fillStyle = '#ffcccc';
                ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
            }
            // Draw Goal
            if (y === 3 && x === COLS - 1) {
                ctx.fillStyle = '#ccffcc';
                ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
            }

            // Draw Policy Arrows
            if (qTable[s]) {
                let bestA = ACTIONS[ACTIONS.map(act => getQ(s, act)).indexOf(Math.max(...ACTIONS.map(act => getQ(s, act))))];
                ctx.fillStyle = "rgba(255, 215, 0, 0.5)";
                ctx.font = "20px Arial";
                let arrow = { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[bestA];
                if (Math.max(...ACTIONS.map(act => getQ(s, act))) !== 0) {
                    ctx.fillText(arrow, x * TILE + 22, y * TILE + 35);
                }
            }
        }
    }
    // Draw Agent
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc(state.x * TILE + TILE/2, state.y * TILE + TILE/2, 15, 0, Math.PI * 2);
    ctx.fill();
}

function resetSim() {
    qTable = {};
    episode = 0;
    state = { x: 0, y: 3 };
    if (!running) { running = true; loop(); }
}

loop();
