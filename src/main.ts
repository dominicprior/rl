// A reinforcement learning demo on the cliff walking grid world.
// You can see it in action here: https://dominicprior.github.io/rl/

// Still under construction...

// Here are some future enhancements.
//
// - A graph of the episode lengths.
// - Split the animation into moving the agent and updating Q.
// - Add a delay for the cliff and the goal.

import './style.css'
type dir = 'LEFT' | 'UP' | 'RIGHT' | 'DOWN';
type pair = [number, number];  // for storing a state
type historyItem = [ pair, dir, number,     // state, action, Q value
          number, number, number, number ]; // stats
type param = [label: string, value: number, min: number, max: number, tip: string];

const TILE = 60;  // tile size

let timeout = 100;
let stopAfterEpisode = true;
// let logging = false;

let params: Record<string, number> = {};

let paramData: Record<string, param> = {

  // Gamma is the discount factor for future rewards.  A value of 1 mean rewards
  // contribute equally to the overall return regardless of when they occur.

  gamma: ['Gamma', 1,   0, 1, 'The discount factor for future rewards'],

  // Epsilon is the probability of exploring randomly, as opposed to greedily
  // choosing the action with the highest Q value.  An epsilon of zero means it
  // always chooses the highest value, instead of exploring, which means Q-Learning
  // and Sarsa are the same.  In this demo, the
  // initial Q values are high (by being zero rather than negative), which
  // makes the agent think every path is amazing until it tries it and is "disappointed."
  // This forces it to traverse almost every arc in the graph at least once
  // before it settles on a favourite.

  epsilon: ['Epsilon', 0,   0, 1, 'How often to explore randomly, as opposed to choosing the best action'],

  // Alpha is the nudge factor for updating a Q value from a successor Q value.

  alpha: ['Alpha', 0.2,   0, 1, 'The learning rate: the proportion of the q-value adjustments towards a target value'],

  goalReward: ['Reward', 10, 0, 20, 'The reward for reaching the goal'],

  cliff_penalty: ['Cliff Penalty', 10,   0, 500, 'The negative reward for falling off the cliff'],
  rows: ['Rows', 4,       1, 10, 'Number of rows'],
  cols: ['Columns', 10,   1, 10, 'Number of columns'],

  init_y: ['Init Y', 3,   0, 10, 'The y-coord (counting from the top from zero) of the initial tile'],
  init_x: ['Init X', 0,   0, 10, 'The x-coord (counting from zero) of the initial tile'],

  cliff_start: ['Cliff Start', 1,   0, 10, 'The x-coord of the first cliff tile'],
  cliff_end:   ['Cliff End',   8,   0, 10, 'The x-coord of the last cliff tile'],

  goal_y: ['Goal Y', 3,   0, 10, 'The y-coord of the goal tile'],
  goal_x: ['Goal X', 9,   0, 10, 'The x-coord of the goal tile'],

};

for (const [id, param] of Object.entries(paramData)) {
  params[id] = param[1];
}

const initialQValueScale = TILE * 0.05;
let qValueScale = initialQValueScale;  // pixels per unit q-value
let maxAbsQValue = 0;

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

let app: HTMLDivElement;
let boxes: HTMLDivElement;
let ctx: CanvasRenderingContext2D;
let canvas: HTMLCanvasElement;

function isCliff(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return y === params.rows - 1 && x >= params.cliff_start && x <= params.cliff_end;
}

function isGoal(pair: pair): boolean {
  const y = pair[0];
  const x = pair[1];
  return y === params.goal_y && x === params.goal_x;
}

function rewind(): void {
  history = [];
  totalSteps = 0, episode = 0, steps = 0, totalReward = 0;
  resetQTable();
  qValueScale = initialQValueScale;
  maxAbsQValue = 0;
  
  state = [params.init_y, params.init_x];
  action = chooseAction(state);
}

// The q-values for state s.  The pair is the y then the x.
function qValues(s: pair): Record<dir, number> {
  return qTable[s[0]][s[1]];
}

// The epsilon-greedy action ('UP', 'DOWN', 'LEFT' or 'RIGHT')
// from the state, s (a string containing the x and y, e.g. '0,3').
function chooseAction(s: pair): dir {
  if (Math.random() < params.epsilon) {
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
function lookupNextState(s: pair, a: dir) {
  let y = s[0];
  let x = s[1];
  if (a === 'UP')    y--;
  if (a === 'DOWN')  y++;
  if (a === 'LEFT')  x--;
  if (a === 'RIGHT') x++;

  let reward = -1;

  if (isCliff([y, x])) {
    reward = -params.cliff_penalty;
    y = params.init_y;
    x = params.init_x;
  }
  else if (isGoal([y, x])) {
    reward = params.goalReward;
  }
  const nextState = [y, x] as pair;
  return { nextState, reward };
}

function step(): boolean {
  history.push([[state[0], state[1]], action, qValues(state)[action],
                  totalSteps, episode, steps, totalReward]);
  let pause = false;
  if (isGoal(state)) {  // if we reached the goal last time
    // do second half of the step - we did the updateQ already
    episode++;
    totalReward = 0;
    steps = 0;
    state = [params.init_y, params.init_x];  // teleport to the start
    action = chooseAction(state);
  }
  else {
    // ordinary step, not second half of another step
    totalSteps++;
    steps++;
    let { nextState, reward } = lookupNextState(state, action);
    totalReward += reward;
    if (isGoal(nextState)) {
      // do the first half of the step - moving onto the goal tile
      nudgeQ(state, action, params.goalReward);  // ??? non-zero epsilon?
      state = nextState;  // but no action
      pause = true;
    }
    else {  // ordinary step that didn't reach the goal
      let nextAction = chooseAction(nextState);
      updateQ(state, action, nextState, nextAction, reward);
      state = nextState;
      action = nextAction;
    }
  }
  draw();
  return pause;  // something to tell scheduleNext whether to schedule or not.
}

function scheduleNext() {
  timeoutId = setTimeout(() => {
    const pause = step() && stopAfterEpisode;
    if (pause) {
      timeoutId = null;
    }
    else {
      scheduleNext();
    }
  }, timeout);
}

function updateQ(s: pair, a: dir, next: pair, nextAction: dir, reward: number): void {
  let algo = (document.getElementById('algoSelect') as HTMLSelectElement).value;
  const q = algo === 'qlearning' ? highestScore(next) : qValues(next)[nextAction];
  let targetValue = reward + params.gamma * q;
  nudgeQ(s, a, targetValue);
  // log(s, a, ' ', next, nextAction, ' ', 'q=' + q, 'target=' + targetValue, 'newQ=' + qValues(s)[a]);
  // log(maxNegQValue);
}

function nudgeQ(s: pair, a: dir, targetValue: number) {
  qValues(s)[a] += params.alpha * (targetValue - qValues(s)[a]);
  maybe_decrease_qValueScale(qValues(s)[a]);
}

function stop_and_rewind() {
  stop_animation();
  rewind();
  draw();
}

function stop_animation() {
  if (timeoutId !== null) {
    clearTimeout(timeoutId);
    timeoutId = null;
  }
}

function resume_animation() {
  stopAfterEpisode = false;
  if (timeoutId === null)
    scheduleNext();
}

function resume_episode() {
  stopAfterEpisode = true;
  if (timeoutId === null)
    scheduleNext();
}

function singleStep() {
  stop_animation();
  step();
}

function back() {
  stop_animation();
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
function maybe_decrease_qValueScale(q: number): void {
  if (Math.abs(q) > maxAbsQValue) {
    maxAbsQValue = Math.abs(q);
    while (maxAbsQValue * qValueScale > 0.9 * TILE) {
      qValueScale /= 2;
    }
  }
}

function draw() {
  ctx.clearRect(0, 0, params.cols * TILE, params.rows * TILE);
  for (let y = 0; y < params.rows; y++) {
    for (let x = 0; x < params.cols; x++) {
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

      if (x === params.init_x && y === params.init_y) {
        ctx.fillStyle = '#ffffbb';
        ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
      if (!isGoal(s) && !isCliff(s)) {
        drawArrow(ctx, s);
      }

      // Draw the four q-values
      if (!isCliff(s) && !isGoal(s)) {
        const q = qValues(s);
        if ('UP' in q) {
          const ww = qSize(q['UP']);
          ctx.fillStyle = qColour(q['UP']);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, y * TILE + 3, ww, 2);
        }
        if ('DOWN' in q) {
          const ww = qSize(q['DOWN']);
          ctx.fillStyle = qColour(q['DOWN']);
          ctx.fillRect(x * TILE + TILE/2 - ww/2, (y+1) * TILE - 4, ww, 2);
        }
        if ('LEFT' in q) {
          const hh = qSize(q['LEFT']);
          ctx.fillStyle = qColour(q['LEFT']);
          ctx.fillRect(x * TILE + 3, y * TILE + TILE/2 - hh/2, 2, hh);
        }
        if ('RIGHT' in q) {
          const hh = qSize(q['RIGHT']);
          ctx.fillStyle = qColour(q['RIGHT']);
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

function qSize(qValue: number): number {
  return Math.abs(qValue) * qValueScale;
}

function qColour(size: number): string {
  return size > 0 ? '#000' : '#f00';
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
    ctx.fillStyle = "#00f";  // if a definite choice
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
  for (let y = 0; y < params.rows; y++) {
    qTable.push([]);
    for (let x = 0; x < params.cols; x++) {
      qTable[y].push({} as Record<dir, number>);
      const q = qTable[y][x];
      if (x > 0) q['LEFT'] = 0;
      if (y > 0) q['UP'] = 0;
      if (x < params.cols - 1) q['RIGHT'] = 0;
      if (y < params.rows - 1) q['DOWN'] = 0;
    }
  }
}

function initdom() {
  app = (document.querySelector<HTMLDivElement>('#app') as HTMLDivElement);

  const heading = document.createElement('h2');
  heading.textContent = 'Cliff Walking Reinforcement Learning';

  const controls = document.createElement('div');
  boxes = document.createElement('div');
  boxes.id = 'boxes';

  const buttons = document.createElement('div');
  const buttons2 = document.createElement('div');
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

  addButton(buttons, 'Faster', () => { timeout /= 2 }, 'Double the frame rate');
  addButton(buttons, 'Slower', () => { timeout *= 2 }, 'Half the frame rate');
  addButton(buttons, 'Pause', stop_animation, 'Pause');
  addButton(buttons, 'Resume episode', resume_episode, 'Continue until the end of the episode (i.e. when we reach the goal)');
  addButton(buttons, 'Resume', resume_animation, 'Continue indefinitely or until paused');
  addButton(buttons, 'Step', singleStep, 'Take one step');
  addButton(buttons, 'Back', back, 'Undo a step');
  addButton(buttons, 'Rewind', stop_and_rewind, 'Rewind all the way to the beginning');
  addButton(buttons2, 'Tiny', () => {
    setSize(1, 3);
  }, 'Start again with a tiny world');
  addButton(buttons2, 'Small', () => {
    setSize(2, 5);
  }, '');
  addButton(buttons2, 'Medium', () => {
    setSize(3, 6);
  }, '');
  addButton(buttons2, 'Large', () => {
    setSize(4, 10);
  }, '');
  // addButton(buttons2, 'Toggle logging', () => { logging = !logging; });

  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.id = 'stats';
  stats.textContent = 'Total Steps: 0 | Episode: 0 | Steps: 0 | Reward: 0';

  const small = document.createElement('small');
  small.innerHTML =
    'Blue blob = Agent | Pink = Cliff | Green = Goal | Arrows = Policy | Black lines = <b>positive</b> Q values | Red lines = <b>negative</b> Q values';

  const srctext = document.createElement('small');
  srctext.innerHTML =
    '<a href="https://github.com/dominicprior/rl/blob/main/src/main.ts">See on GitHub</a>';

  controls.appendChild(buttons);
  controls.appendChild(buttons2);
  controls.appendChild(select);
  controls.appendChild(stats);
  controls.appendChild(small);
  for (const [id, param] of Object.entries(paramData)) {
    addInput(boxes, id, param);
  }

  app.appendChild(heading);
  app.appendChild(controls);
  app.appendChild(boxes);
  createCanvas();
  app.appendChild(srctext);
}

function setSize(rows: number, cols: number) {
  setParam('rows', rows);
  setParam('cols', cols);
  setParam('init_y', rows - 1);
  setParam('init_x', 0);
  setParam('goal_y', rows - 1);
  setParam('goal_x', cols - 1);
  setParam('cliff_start', 1);
  setParam('cliff_end', rows === 1 ? 0 : cols - 2);
  createCanvas();
  stop_and_rewind();
}

function createCanvas() {
  if (canvas) {
    canvas.remove();
  }
  canvas = (document.createElement('canvas') as HTMLCanvasElement);
  canvas.width  = params.cols * TILE;
  canvas.height = params.rows * TILE;
  boxes.after(canvas);
  ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
}

function addInput(controls: HTMLDivElement, id: string, param: param) {
  if (['rows', 'init_y', 'goal_y'].includes(id)) {
    controls.appendChild(document.createElement('div'));
  }
  controls.appendChild(createNumberInput(id, param))
}

function createNumberInput(id: string, param: param) {
  const [label, value, min, max, tip] = param;
  params[id] = value;
  const wrapper = document.createElement('span');
  wrapper.setAttribute('class', 'tooltip');
  wrapper.setAttribute('data-tip', tip);
  wrapper.style.margin = '10px';

  const lbl = document.createElement('label');
  lbl.textContent = ' ' + label + ' ';
  lbl.htmlFor = id;

  const input = document.createElement('input');
  input.type = 'number';
  input.id = id;
  input.min = '' + min;
  input.max = '' + max;
  // input.step = step ?? 1;
  input.value = '' + value;

  input.addEventListener('change', () => {
    const val = parseFloat(input.value);
    if (!isNaN(val)) {
      params[input.id] = val;
    }
    console.log(input.id, val);
    if (['rows', 'cols'].includes(input.id)) {
      console.log('changing size');
      clampParams();
      createCanvas();
      stop_and_rewind();
    }
  })

  wrapper.appendChild(lbl);
  wrapper.appendChild(input);
  return wrapper;
}

function clampParams() {
  clamp('init_y', 'rows');
  clamp('goal_y', 'rows');
  clamp('goal_x', 'cols');
}

function clamp(param: string, limit: string) {
  if (params[param] >= params[limit]) {
    setParam(param, params[limit] - 1);
  }
}

function setParam(param: string, value: number) {
  params[param] = value;
  (document.getElementById(param) as HTMLInputElement).value = '' + value;
}

function addButton(controls: HTMLDivElement, text: string, f: () => void, tip: string) {
  const b = document.createElement('button');
  b.textContent = text;
  b.addEventListener('click', f);
  b.setAttribute('class', 'tooltip');
  b.setAttribute('data-tip', tip);
  controls.appendChild(b);
}

// function log(...args: any[]): void {
//   if (logging) {
//     console.log(...args);
//   }
// }

initdom();
rewind();
scheduleNext();
