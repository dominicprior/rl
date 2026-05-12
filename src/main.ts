// A reinforcement learning demo on the cliff walking grid world.
// You can see it in action here: https://dominicprior.github.io/rl/

// Still under construction...

// Here are some future enhancements.
//
// - Optional cliff.
// - Variable gamma, epsilon and alpha.
// - A graph of the episode lengths.
// - Split the animation into moving the agent and updating Q.
// - Add a delay for the cliff and the goal.

import './style.css'
type dir = 'LEFT' | 'UP' | 'RIGHT' | 'DOWN';
type pair = [number, number];  // for storing a state
type historyItem = [ pair, dir, number,     // state, action, Q value
          number, number, number, number ]; // stats
type param = [label: string, value: number, min: number, max: number];
const ROWS = 4, COLS = 10;
const TILE = 60;  // tile size
const w = COLS * TILE;
const h = ROWS * TILE;
const thereIsACliff = true;

let timeout = 100;
let logging = false;

let paramData: Record<string, param> = {
  speed: ['Speed!', 50, 100, 0],
};
let params: Record<string, number> = {};

// let params: Array<param> = [
//   ['speed', 'Speed!', 50, 100, 0],
// ];

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

let qValueScale = TILE;  // pixels per unit q-value
let maxNegQValue = 0; // apart from ones bordering the cliff

const INIT_Y = ROWS - 1, INIT_X = 0;
const GOAL_Y = ROWS - 1, GOAL_X = COLS - 1;

// The qTable is an array of rows.
// Each row is an array of cells.
// Each cell is a mapping from direction to q-value.
// y first again.
let qTable: Array<Array<Record<dir, number> > >;  // e.g. qTable[0][0]['DOWN'] would be a Q value.

let state: pair;  // y first, i.e. row first
let action: dir;

let timeoutId: number | null = null;

let totalSteps = 0, episode = 0, steps = 0, totalReward = 0;

let history: Array<historyItem>;

const ctx = initdom();

function isCliff(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return thereIsACliff && (y === ROWS - 1 && x > 0 && x < COLS - 1);
}

function isGoal(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return y === GOAL_Y && x === GOAL_X;
}

function resetGlobals(): void {
  history = [];
  totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
  resetQTable();
  state = [INIT_Y, INIT_X];
  action = chooseAction(state);
}

// The q-values for state s.  The pair is the y then the x.
function qValues(s: pair): Record<dir, number> {
  return qTable[s[0]][s[1]];
}

// The epsilon-greedy action ('UP', 'DOWN', 'LEFT' or 'RIGHT')
// from the state, s (a string containing the x and y, e.g. '0,3').
function chooseAction(s: pair): dir {
  if (Math.random() < epsilon) {
    const aa = actions(s);
    return aa[Math.floor(Math.random() * aa.length)];
  }
  else {
    return highestAction(s);
  }
}

// The action with the highest Q value.
// (In the event of a tie choose the first action).
function highestAction(s: pair): dir {
  let ss = scores(s);
  let i = ss.indexOf(highestScore(s));  // 0, 1, 2 or 3
  return actions(s)[i];  // 'UP', 'DOWN', 'LEFT' or 'RIGHT'
}

// The highest Q value from the state, s.
function highestScore(s: pair): number {
  return Math.max(...scores(s));
}

// The possible actions from the state, s.
function actions(s: pair): Array<dir> {
  return Object.keys(qValues(s)) as Array<dir>;
}

// The Q values from the state, s.
function scores(s: pair): Array<number> {
  return Object.values(qValues(s));
}

// The result of taking action a from state s.
function calcNextState(s: pair, a: dir) {
  let y = s[0];
  let x = s[1];
  if (a === 'UP')    y--;
  if (a === 'DOWN')  y++;
  if (a === 'LEFT')  x--;
  if (a === 'RIGHT') x++;

  let reward = -1;
  let done = false;

  if (isCliff([y, x])) {
    reward = -100;
    y = INIT_Y;
    x = INIT_X;
  }
  else if (isGoal([y, x])) {
    reward = 0;
    episode++;
    steps = 0;
    totalReward = 0;
    y = INIT_Y;
    x = INIT_X;
    done = true;
  }
  const nextState = [y, x] as pair;
  return { nextState, reward, done };
}

function updateQ(s: pair, a: dir, next: pair, nextAction: dir, reward: number, done: boolean): void {
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  const q = done ? 0 :
              algo === 'qlearning' ? highestScore(next) : qValues(next)[nextAction];
  let target = reward + gamma * q;
  qValues(s)[a] += alpha * (target - qValues(s)[a]);
  maybe_decrease_qValueScale(qValues(s)[a], reward);
  log(s, a, ' ', next, nextAction, ' ', 'q=' + q, 'target=' + target, 'newQ=' + qValues(s)[a]);
  log(maxNegQValue);
}

function step(): void {
  let { nextState, reward, done } = calcNextState(state, action);
  let nextAction = chooseAction(nextState);
  history.push([[state[0], state[1]], action, qValues(state)[action],
                  totalSteps, episode, steps, totalReward]);
  totalReward += reward;
  totalSteps++;
  steps++;
  updateQ(state, action, nextState, nextAction, reward, done);
  state = nextState;
  action = nextAction;
  draw();
  return;
}

function scheduleNext() {
  timeoutId = setTimeout(() => {
    step();
    scheduleNext();
  }, timeout);
}

function reset() {
  pause();
  resetGlobals();
  draw();
}

function pause() {
  if (timeoutId !== null) {
    clearTimeout(timeoutId);
    timeoutId = null;
  }
}

function resume() {
  if (timeoutId === null) scheduleNext();
}

function singleStep() {
  pause();
  step();
}

function back() {
  pause();
  if (history.length !== 0) {
    const h = history.pop() as historyItem;
    state  = h[0];
    action = h[1];
    [qValues(state)[action],
        totalSteps, episode, steps, totalReward ] = (h.slice(2) as Array<number>);
    draw();
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
      let s = [y, x] as pair;
      ctx.strokeStyle = '#ccc';
      ctx.strokeRect(x * TILE, y * TILE, TILE, TILE);

      if (isCliff(s)) {
        ctx.fillStyle = '#ffcccc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }

      if (isGoal(s)) {
        ctx.fillStyle = '#ccffcc';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }

      drawArrow(ctx, s);

      // Draw the four q-values
      if (!isCliff(s) && !isGoal(s)) {
        ctx.fillStyle = '#000000';
        const q = qValues(s);
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
  // Draw Agent
  ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
  ctx.beginPath();
  ctx.arc(state[1] * TILE + TILE/2, state[0] * TILE + TILE/2, TILE/4, 0, Math.PI * 2);
  ctx.fill();
  // Write stats
  (document.getElementById('stats') as HTMLDivElement).innerText =
      `Total Steps: ${totalSteps} | Episode: ${episode} | Steps: ${steps} | Reward: ${totalReward}`;
}

function drawArrow(ctx: CanvasRenderingContext2D, s: pair) {
  const y = s[0], x = s[1];
  const q = qValues(s);  // e.g. { DOWN: 3, RIGHT: 4 }
  // Find the indexes of the actions in the reverse order of their Q values.
  const aa: Array<dir> = actions(s).toSorted(  // e.g. ['RIGHT', 'DOWN']
    (a,b) => q[b] - q[a]
  );

  const bestA = aa[0];  // e.g. 'UP'
  ctx.font = `${TILE/3}px Arial`;
  // ctx.font = "20px Arial";
  const str = arrowStr(bestA);
  if (q[aa[1]] === 0) {
    ctx.fillStyle = "#ddd";  // if still a tie
  }
  else {
    ctx.fillStyle = "#f80";  // if a definite choice
  }
  const isThin = bestA === 'UP' || bestA === 'DOWN';
  ctx.fillText(str, (x + (isThin ? 0.42 : 0.35)) * TILE,
        (y + 0.59) * TILE);
}

function arrowStr(dir: dir): string {
  return { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' }[dir];
}

function resetQTable() {
  qTable = [];
  for (let y = 0; y < ROWS; y++) {
    qTable.push([]);
    for (let x = 0; x < COLS; x++) {
      qTable[y].push({} as Record<dir, number>);
      const q = qTable[y][x];
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

  const select = document.createElement('select');
  select.id = 'algoSelect';

  const options = [
    { value: 'qlearning', label: 'Q-Learning (Off-policy)' },
    { value: 'sarsa', label: 'SARSA (On-policy)' },
  ];

  for (const opt of options) {
    const optionEl = document.createElement('option');
    optionEl.value = opt.value;
    optionEl.textContent = opt.label;
    select.appendChild(optionEl);
  }

  // addButton(controls, 'Reset', () => {});
  addButton(controls, 'Faster', () => { timeout /= 2 });
  addButton(controls, 'Slower', () => { timeout *= 2 });
  addButton(controls, 'Toggle logging', () => { logging = !logging; });
  addButton(controls, 'Pause', pause);
  addButton(controls, 'Resume', resume);
  addButton(controls, 'Step', singleStep);
  addButton(controls, 'Back', back);
  addButton(controls, 'Reset', reset);

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
  controls.appendChild(stats);
  controls.appendChild(small);
  for (const [id, param] of Object.entries(paramData)) {
    addInput(controls, id, param);
  }

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

function addInput(controls: HTMLDivElement, id: string, param: param) {
  controls.appendChild(createNumberInput(id, param))
}

function createNumberInput(id: string, param: param) {
  const [label, value, min, max] = param;
  params[id] = value;
  const wrapper = document.createElement('div');

  const lbl = document.createElement('label');
  lbl.textContent = label;
  lbl.htmlFor = id;

  const input = document.createElement('input');
  input.type = 'number';
  input.id = id;
  input.min = '' + min;
  input.max = '' + max;
  // input.step = step ?? 1;
  input.value = '' + value;

  input.addEventListener('change', (event) => {
    const val = parseFloat(input.value);
    if (!isNaN(val)) {
      params[input.id] = val;
    }
    console.log(event);
    console.log(input);
    console.log(input.value);
    console.log(params.speed);
  })

  // input.addEventListener('input', () => {
  //   const val = parseFloat(input.value);
  //   if (!isNaN(val)) onchange(val);
  // });

  wrapper.appendChild(lbl);
  wrapper.appendChild(input);
  return wrapper;
}

function addButton(controls: HTMLDivElement, text: string, f: () => void) {
  const b = document.createElement('button');
  b.textContent = text;
  b.addEventListener('click', f);
  controls.appendChild(b);
}

function log(...args: any[]): void {
  if (logging) {
    console.log(...args);
  }
}

resetGlobals();
scheduleNext();
