// A reinforcement learning demo on the cliff walking grid world.
// You can see it in action here: https://dominicprior.github.io/rl/

// Still under construction...

// Here are some future enhancements.
//
// - Pause and single-step buttons.
// - A way to rewind the demo.
// - Earlier arrows.
// - Optional cliff.
// - Fix Monte Carlo.
// - Variable w and h.
// - Variable gamma, epsilon and alpha.
// - A graph of the episode lengths.
// - Split the animation into moving the agent and updating Q.
// - Add a delay for the cliff and the goal.

import './style.css'
const w = 600;
const h = 240;
let timeout = 100;
let logging = false;

// Gamma is the discount factor for future rewards.  A value of 1 mean rewards
// contribute equally to the overall return regardless of when they occur.
let gamma = 1;

// Epsilon is the probability of exploring randomly, as opposed to greedily
// choosing the action with the highest Q value.  An epsilon of zero means it
// always chooses the highest, instead of exploring, which means Q-Learning
// and Sarsa are the same.  In this demo, the
// initial Q values are high (by being zero rather than negative), which
// makes the agent think every path is amazing until it tries it and is "disappointed."
// This forces it to traverse almost every arc in the graph at least once
// before it settles on a favourite.
let epsilon = 0;

// The nudge factor for updating a Q value from a successor Q value.
let alpha = 0.1;

const ctx = initdom();

const ROWS = 4, COLS = 10, TILE = 60;
let qValueScale = TILE;  // pixels per unit q-value
let maxNegQValue = 0; // apart from ones bordering the cliff

let state = { x: 0, y: 3 };
let qTable: Record<string, Record<string, number> >;  // e.g. qTable['0,0']['DOWN'] would be a Q value.
resetQTable();

let totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
let resetting = false;
let history: Array<any> = []; // For Monte Carlo  // not working any more

// The epsilon-greedy action ('UP', 'DOWN', 'LEFT' or 'RIGHT')
// from the state, s (a string containing the x and y, e.g. '0,3').
function action(s: string): string {
  if (Math.random() < epsilon) {
    const actions = Object.keys(qTable[s]);
    return actions[Math.floor(Math.random() * actions.length)];
  }
  else {
    return highestAction(s);
  }
}

// The action with the highest Q value.
// (In the event of a tie choose the first action).
function highestAction(s: string): string {
  let ss = scores(s);
  let i = ss.indexOf(highestScore(s));  // 0, 1, 2 or 3
  return actions(s)[i];  // 'UP', 'DOWN', 'LEFT' or 'RIGHT'
}

// The highest Q value from the state, s.
function highestScore(s: string): number {
  return Math.max(...scores(s));
}

// The possible actions from the state, s.
function actions(s: string): Array<string> {
  return Object.keys(qTable[s]);
}

// The Q values from the state, s.
function scores(s: string): Array<number> {
  return Object.values(qTable[s]);
}

// The result of taking action a from state s.
function nextState(s: {x: number, y: number}, a: string) {
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
  let a = action(s);
  
  while (true) {
    let { next, reward, done } = nextState(state, a);  // calc the next state
    let sNext = `${next.x},${next.y}`;
    let aNext = action(sNext);  // what the action would be from that next state
    
    if (algo !== 'montecarlo') {
      const q = algo === 'qlearning' ? highestScore(sNext) : qTable[sNext][aNext];
      let target = reward + gamma * q;
      qTable[s][a] += alpha * (target - qTable[s][a]);
      maybe_decrease_qValueScale(qTable[s][a], reward);
      log(s, a, ' ', sNext, aNext, ' ', 'q=' + q, 'target=' + target, 'newQ=' + qTable[s][a]);
      log(maxNegQValue);
    }
    else {
      history.push({ s, a, r: reward });
    }

    state = next;
    s = sNext;
    a = aNext;
    steps++;
    totalSteps++;
    totalReward += reward;

    draw();
    (document.getElementById('stats') as HTMLDivElement).innerText =
      `Total Steps: ${totalSteps} | Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;

    if (resetting) {
      totalSteps = 0, episode = 0, steps = 0, totalReward = 0, maxNegQValue = 0;
      state = { x: 0, y: 3 };
      s = '0,3';
      a = action(s);
      resetQTable();
      resetting = false;
    }
    else if (done || steps > 500) {
      if (algo === 'montecarlo') {  // not working any more
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
      a = action(s);
    }
    await new Promise(r => setTimeout(r, timeout));
  }
}

// Make sure the q-values lines fit inside the tiles.
function maybe_decrease_qValueScale(q: number, reward: number): void {
  if (-q > maxNegQValue && reward !== -100) {
    maxNegQValue = -q;
    if (maxNegQValue * qValueScale > 0.9 * TILE) {
      qValueScale /= 2;
    }
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
        const max = highestScore(s);    // e.g. -1.6
        const bestA = highestAction(s);  // e.g. 2
        ctx.fillStyle = "#f80";
        ctx.font = "20px Arial";
        const arrow = ({ UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[bestA] as string);
        if (max !== 0) {
            ctx.fillText(arrow, x * TILE + 22, y * TILE + 35);
        }

        // Draw the four q-values
        if (y !== 3 || x === 0) {
          ctx.fillStyle = '#000000';
          const q = qTable[s];
          if ('UP' in q) {
            const ww = Math.min(-q['UP'] * qValueScale, TILE - 4);
            ctx.fillRect(x * TILE + TILE/2 - ww/2, y * TILE + 3, ww, 2);
          }
          if ('DOWN' in q) {
            const ww = Math.min(-q['DOWN'] * qValueScale, TILE - 4);
            ctx.fillRect(x * TILE + TILE/2 - ww/2, (y+1) * TILE - 4, ww, 2);
          }
          if ('LEFT' in q) {
            const hh = Math.min(-q['LEFT'] * qValueScale, TILE - 4);
            ctx.fillRect(x * TILE + 3, y * TILE + TILE/2 - hh/2, 2, hh);
          }
          if ('RIGHT' in q) {
            const hh = Math.min(-q['RIGHT'] * qValueScale, TILE - 4);
            ctx.fillRect((x+1) * TILE - 4, y * TILE + TILE/2 - hh/2, 2, hh);
          }
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
  resetting = true;
}

function resetQTable() {
  qTable = {};
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      let s = `${x},${y}`;
      qTable[s] = {};
      const q = qTable[s];
      if (x > 0) q['LEFT'] = 0;
      if (y > 0) q['UP'] = 0;
      if (x < COLS - 1) q['RIGHT'] = 0;
      if (y < ROWS - 1) q['DOWN'] = 0;
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
    // { value: 'montecarlo', label: 'Monte Carlo (Every-visit)' },  // not working any more
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
  stats.textContent = 'Total Steps: 0 | Episode: 0 | Steps: 0 | Reward: 0';

  const small = document.createElement('small');
  small.textContent =
    'Blue = Agent | Pink = Cliff | Green = Goal | Orange arrows = Policy | Black lines = Q values';

  const srctext = document.createElement('small');
  srctext.innerHTML =
    '<a href="https://github.com/dominicprior/rl/blob/main/src/main.ts">See on GitHub</a>';

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
  app.appendChild(srctext);
  return (canvas.getContext('2d') as CanvasRenderingContext2D);
}

function log(...args: any[]): void {
  if (logging) {
    console.log(...args);
  }
}

draw();
await new Promise(r => setTimeout(r, timeout));
loop();
