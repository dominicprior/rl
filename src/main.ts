import './style.css'

const w = 600;
const h = 240;
let timeout = 1000;
let logging = false;
let gamma = 0.9;
let epsilon = 0.1;

const ctx = initdom();

const ROWS = 4, COLS = 10, TILE = 60;
const ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

let state = { x: 0, y: 3 };
let qTable: Record<string, Record<string, number> >;
resetQTable();

let episode = 0, steps = 0, totalReward = 0;
let running = true;
let history: Array<any> = []; // For Monte Carlo

function chooseAction(s: string) {
  if (Math.random() < epsilon) return ACTIONS[Math.floor(Math.random() * 4)];
  let scores = ACTIONS.map(a => qTable[s][a]);
  return ACTIONS[scores.indexOf(Math.max(...scores))];
}

function move(s: {x: number, y: number}, a: string) {
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
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  let s = `${state.x},${state.y}`;
  let a = chooseAction(s);
  
  while (running) {
    let { next, reward, done } = move(state, a);
    let sNext = `${next.x},${next.y}`;
    let aNext = chooseAction(sNext);

    if (algo === 'qlearning') {
      let maxNextQ = Math.max(...ACTIONS.map(act => qTable[sNext][act]));
      qTable[s][a] += 0.1 * (reward + gamma * maxNextQ - qTable[s][a]);
      if (logging) {
        console.log(s, a, qTable[s][a]);
      }
    } else if (algo === 'sarsa') {
      qTable[s][a] += 0.1 * (reward + gamma * qTable[sNext][aNext] - qTable[s][a]);
    } else if (algo === 'montecarlo') {
      history.push({ s, a, r: reward });
    }

    state = next;
    s = sNext;
    a = aNext;
    steps++;
    totalReward += reward;

    draw();
    (document.getElementById('stats') as HTMLDivElement).innerText = `Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;

    if (done || steps > 500) {
      if (algo === 'montecarlo') {
        let G = 0;
        for (let i = history.length - 1; i >= 0; i--) {
          G = history[i].r + gamma * G;
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
    await new Promise(r => setTimeout(r, timeout));
  }
}

function draw() {
  ctx.clearRect(0, 0, w, h);
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

      if (qTable[s]) {
        // Draw a policy arrow
        const qValues = ACTIONS.map(act => qTable[s][act]);  // qValues[0] == UP q-value
        const max = Math.max(...qValues);    // e.g. -1.6
        const bestI = qValues.indexOf(max);  // e.g. 2
        const bestA = ACTIONS[bestI];        // e.g. 'LEFT'
        ctx.fillStyle = "rgba(0, 0, 0, 1)";
        ctx.font = "20px Arial";
        const arrow = ({ UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[bestA] as string);
        if (max !== 0) {
            ctx.fillText(arrow, x * TILE + 22, y * TILE + 35);
        }

        // Draw the four q-values
        if (y !== 3 || x === 0) {
          ctx.fillStyle = '#000000';
          let ww = Math.min(-qTable[s]['UP'] * 20, TILE - 4);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, y * TILE + 3, ww, 4);
          ww = Math.min(-qTable[s]['DOWN'] * 20, TILE - 4);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, (y+1) * TILE - 6, ww, 4);
          let hh = Math.min(-qTable[s]['LEFT'] * 20, TILE - 4);
          ctx.fillRect(x * TILE + 3, y * TILE + TILE/2 - hh/2, 4, hh);
          hh = Math.min(-qTable[s]['RIGHT'] * 20, TILE - 4);
          ctx.fillRect((x+1) * TILE - 6, y * TILE + TILE/2 - hh/2, 4, hh);
        }
      }
    }
  }
  // Draw Agent
  ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
  ctx.beginPath();
  ctx.arc(state.x * TILE + TILE/2, state.y * TILE + TILE/2, 15, 0, Math.PI * 2);
  ctx.fill();
}

function resetSim() {
  resetQTable();
  episode = 0;
  state = { x: 0, y: 3 };
  if (!running) { running = true; loop(); }
}

function resetQTable() {
  qTable = {};
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let s = `${x},${y}`;
      qTable[s] = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
    }
  }
}

function initdom() {
  const app = (document.querySelector<HTMLDivElement>('#app') as HTMLDivElement);

  const heading = document.createElement('h2');
  heading.textContent = 'Cliff Walking Reinforcement Learning';

  const controls = document.createElement('div');
  controls.className = 'controls';

  controls.append('Algorithm: ');

  const select = document.createElement('select');
  select.id = 'algoSelect';

  const options = [
    { value: 'qlearning', label: 'Q-Learning (Off-policy)' },
    { value: 'sarsa', label: 'SARSA (On-policy)' },
    { value: 'montecarlo', label: 'Monte Carlo (Every-visit)' },
  ];

  for (const opt of options) {
    const optionEl = document.createElement('option');
    optionEl.value = opt.value;
    optionEl.textContent = opt.label;
    select.appendChild(optionEl);
  }

  const button = document.createElement('button');
  button.textContent = 'Reset & Run';
  button.addEventListener('click', () => {
    resetSim();
  });

  const fastButton = document.createElement('button');
  fastButton.textContent = 'Faster';
  fastButton.addEventListener('click', () => {
    timeout /= 2;
  });

  const slowButton = document.createElement('button');
  slowButton.textContent = 'Slower';
  slowButton.addEventListener('click', () => {
    timeout *= 2;
  });

  const loggingButton = document.createElement('button');
  loggingButton.textContent = 'Toggle logging';
  loggingButton.addEventListener('click', () => {
    logging = !logging;
  });

  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.id = 'stats';
  stats.textContent = 'Episode: 0 | Steps: 0 | Reward: 0';

  const small = document.createElement('small');
  small.textContent =
    'Blue = Agent | Red = Cliff | Green = Goal | Yellow arrows = Policy';

  controls.appendChild(select);
  controls.appendChild(button);
  controls.appendChild(fastButton);
  controls.appendChild(slowButton);
  controls.appendChild(loggingButton);
  controls.appendChild(stats);
  controls.appendChild(small);

  const canvas = (document.createElement('canvas') as HTMLCanvasElement);
  canvas.id = 'gridCanvas';
  canvas.width = w;
  canvas.height = h;

  app.appendChild(heading);
  app.appendChild(controls);
  app.appendChild(canvas);
  return (canvas.getContext('2d') as CanvasRenderingContext2D);
}

loop();
