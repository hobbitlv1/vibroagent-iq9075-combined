// Run with node; exercises the production polling functions without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/vibroagent_mcp/static/vibrogemma/vibroagent.js'), 'utf8');
const finite = new Function(source.match(/  const finite =[^\n]+/)[0] + '; return finite;')();
assert.equal(finite(null), null);
assert.equal(finite(undefined), null);
assert.equal(finite(''), null);
assert.equal(finite(0), 0);
const body = source.slice(source.indexOf('  async function fetchWaveforms()'), source.indexOf('  function waveformAmplitude('));
const targetBody = source.slice(source.indexOf('  function isSharedTargetPattern('), source.indexOf('  function renderMonitorScaffold('));
const state = { paused: false, view: 'monitor', sensors: [{ sensor_id: 'target_1' }], readings: new Map(), monitorAxis: 'x', waveformGeneration: 0 };
const pending = [];
const timers = new Map();
let lastStatus;
let nextTimer = 0;
const api = (url, options) => new Promise(resolve => pending.push({ url, options, resolve }));
const statusNode = () => {
  const classes = new Set();
  return {classList: {
    toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
    contains(name) { return classes.has(name); },
  }};
};
const baselineWaveStatus = {};
const baselineBoardStatus = {};
const baselineRow = {...statusNode(), querySelector: () => baselineWaveStatus};
const baselineRibbon = statusNode();
const elements = {
  '#waveformStack': {children: [baselineRow]},
  '#boardStates': {children: [{querySelector: () => baselineBoardStatus}]},
  '#verdictRibbon': {children: [baselineRibbon]},
};
const functions = new Function('state', 'api', 'waveformUrl', 'redrawWaveforms', 'setLiveState', 'setTimeout', 'clearTimeout', 'AbortController', '$', 'boardLabel',
  targetBody + body + '; return {fetchWaveforms, refreshWaveforms, updateLiveCount, targetState};')(
  state, api, () => state.monitorAxis, () => {}, (ok, text) => { lastStatus = { ok, text }; },
  callback => { timers.set(++nextTimer, callback); return nextTimer; }, id => timers.delete(id), AbortController,
  selector => elements[selector], () => 'Reference');
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const old = functions.fetchWaveforms();
  await functions.fetchWaveforms();
  assert.equal(pending.length, 1, 'timer polls must not queue overlapping SDK requests');
  state.monitorAxis = 'y';
  functions.refreshWaveforms();
  assert(pending[0].options.signal.aborted);
  pending[0].resolve({ axis: 'x', samples_g: [1], metadata: { data_is_current: true } });
  await old;
  assert.equal(state.readings.size, 0, 'late X response must not replace Y');
  assert.equal(pending.length, 2);
  pending[1].resolve({ axis: 'y', samples_g: [2], metadata: { data_is_current: true } });
  await tick();
  assert.equal(state.readings.get('target_1').axis, 'y');
  assert.equal(timers.size, 0);

  const request = functions.fetchWaveforms();
  const timeout = [...timers.values()][0];
  timeout();
  assert(pending[2].options.signal.aborted, 'slow requests have a bounded timeout');
  pending[2].resolve({ error: 'timeout' });
  await request;
  assert(!lastStatus.ok);

  state.sensors = Array.from({ length: 6 }, (_, i) => ({ sensor_id: `board_${i}` }));
  state.readings.clear();
  functions.updateLiveCount();
  assert.equal(lastStatus.text, 'Waiting · 0/6 boards');
  state.sensors.forEach(({ sensor_id }) => state.readings.set(sensor_id, { samples_g: [1], receivedAt: Date.now(), metadata: { data_is_current: true } }));
  functions.updateLiveCount();
  assert(lastStatus.ok && lastStatus.text === 'Live · 6/6 boards');
  state.readings.get('board_0').error = 'live_data_required';
  functions.updateLiveCount();
  assert(!lastStatus.ok && lastStatus.text === 'Waiting · 5/6 boards');
  state.readings.forEach(reading => { reading.receivedAt -= 11000; });
  functions.updateLiveCount();
  assert.equal(lastStatus.text, 'Waiting · 0/6 boards');
  state.baselineId = 'board_0';
  state.gemmaTargets = [{sensor_id: 'board_1', class: 'data_invalid'}];
  const expectBaseline = (label, kind) => {
    functions.updateLiveCount();
    assert.equal(baselineWaveStatus.textContent, label);
    assert.equal(baselineBoardStatus.textContent, label);
    assert.equal(baselineWaveStatus.className, `wave-status ${kind}`);
    assert.equal(baselineBoardStatus.className, `state-value ${kind}`);
    assert(baselineRow.classList.contains(kind));
    assert(baselineRibbon.classList.contains(kind));
    assert.equal(baselineRibbon.title, `Reference · ${label}`);
  };
  state.readings.delete('board_0');
  expectBaseline('Waiting', 'unavailable');
  const baselineReading = {samples_g: [1], receivedAt: Date.now(), metadata: {data_is_current: true}};
  state.readings.set('board_0', baselineReading);
  expectBaseline('Live', 'normal');
  baselineReading.receivedAt -= 11000;
  expectBaseline('Waiting', 'unavailable');
  baselineReading.receivedAt = Date.now();
  baselineReading.metadata.data_is_current = false;
  expectBaseline('Waiting', 'unavailable');
  baselineReading.metadata.data_is_current = true;
  baselineReading.error = 'live_data_required';
  expectBaseline('Waiting', 'unavailable');
  delete baselineReading.error;
  baselineReading.samples_g = [];
  expectBaseline('Waiting', 'unavailable');
  baselineReading.samples_g = [1];
  expectBaseline('Live', 'normal');
  state.offlineReplay = {position_s: 15};
  expectBaseline('Replay', 'normal');
  assert.deepEqual(functions.targetState('board_1'), {label: 'Data invalid', kind: 'quality'},
    'reference source labels must not change target model verdicts');
  state.offlineReplay = null;
  const monitorBody = source.slice(source.indexOf('  async function fetchMonitor()'), source.indexOf('  function renderMonitorResult('));
  let resolveMonitor;
  state.injectionGeneration = 0;
  state.gemmaTargets = [];
  const fetchMonitor = new Function('state', 'api', 'monitorContext', 'renderMonitorResult', 'renderContext', 'hideAlert',
    monitorBody + '; return fetchMonitor;')(state, () => new Promise(resolve => { resolveMonitor = resolve; }),
    () => ({ available: true, targets: ['old-target-5'] }), () => {}, () => {}, () => {});
  const monitorRequest = fetchMonitor();
  state.injectionGeneration += 1;
  resolveMonitor({});
  await monitorRequest;
  assert.deepEqual(state.gemmaTargets, [], 'old monitor response must not reappear after an injection change');
  const streamBody = source.slice(source.indexOf('  function renderStreamStatus()'), source.indexOf('  function renderSensorControls()'));
  const streamPending = [];
  const streamFunctions = new Function('state', 'finite', 'api', 'setLiveState', 'setTimeout', 'clearTimeout', 'AbortController', 'console',
    streamBody + '; return {renderStreamStatus, refreshStreamStatus};')(
    state, finite, (url, options) => new Promise((resolve, reject) => streamPending.push({url, options, resolve, reject})),
    (ok, text) => { lastStatus = {ok, text}; },
    callback => { timers.set(++nextTimer, callback); return nextTimer; }, id => timers.delete(id), AbortController, {warn() {}});
  state.streamStatus = null;
  streamFunctions.renderStreamStatus();
  assert.equal(lastStatus.text, 'Stream status unavailable');
  const streamRequest = streamFunctions.refreshStreamStatus();
  await streamFunctions.refreshStreamStatus();
  assert.equal(streamPending.length, 1, 'acquisition-status polls cannot overlap');
  assert.equal(streamPending[0].url, '/api/vibro/sensors?process_reader=0&prewarm=0');
  const sensors = state.sensors.map(sensor => ({...sensor,
    stream: {data_is_current: true, data_file_age_s: 0.1, data_currentness_max_age_s: 3}}));
  streamPending[0].resolve({sensors});
  await streamRequest;
  assert(lastStatus.ok && lastStatus.text === 'Live · 6/6 boards');
  sensors[2].stream.data_is_current = false;
  streamFunctions.renderStreamStatus();
  assert.equal(lastStatus.text, 'Waiting · 5/6 boards');
  sensors[0].stream.data_file_age_s = null;
  streamFunctions.renderStreamStatus();
  assert.equal(lastStatus.text, 'Waiting · 4/6 boards', 'missing ages cannot count as fresh');
  state.streamStatusAt -= 5000;
  streamFunctions.renderStreamStatus();
  assert.equal(lastStatus.text, 'Waiting · 0/6 boards', 'cached status expires without a new response');
  const failedStatus = streamFunctions.refreshStreamStatus();
  [...timers.values()][0]();
  assert(streamPending[1].options.signal.aborted, 'status requests have a bounded timeout');
  streamPending[1].reject(new Error('timeout'));
  await failedStatus;
  assert.equal(lastStatus.text, 'Stream status unavailable');
  assert.equal(state.streamStatusRequest, null);
  assert.equal(timers.size, 0);
  state.offlineReplay = {position_s: 15};
  functions.updateLiveCount();
  assert.equal(lastStatus.text, 'Offline replay · 6/6 recordings');
  streamFunctions.renderStreamStatus();
  assert.equal(lastStatus.text, 'Offline replay · 6/6 recordings');
  const replayState = {replays: [], replayIndex: -1, replay: null};
  const loadReplayBody = source.slice(source.indexOf('  async function loadReplays('), source.indexOf('  const replayStateLabel'));
  const selectReplayBody = source.slice(source.indexOf('  async function selectReplay('), source.indexOf('  function renderReplay()'));
  const replayRequests = [];
  let renderedReplay;
  let selectedReplayId;
  const replayFunctions = new Function('state', 'api', 'renderReplayList', 'renderReplay', '$',
    loadReplayBody + selectReplayBody + '; return {loadReplays, selectReplay};')(replayState,
    url => new Promise(resolve => replayRequests.push({url, resolve})),
    () => { selectedReplayId = replayState.replays[replayState.replayIndex]?.window_id; },
    () => { renderedReplay = replayState.replay; }, () => ({}));
  const first = {window_id: 'first'};
  const second = {window_id: 'second'};
  replayState.replays = [first, second];
  const pendingSecond = replayFunctions.selectReplay(1);
  const refresh = replayFunctions.loadReplays(true);
  replayRequests[1].resolve({windows: [{window_id: 'newest'}, first, second]});
  await refresh;
  assert.equal(selectedReplayId, 'second', 'refresh retains a non-first selection while its detail is pending');
  assert.equal(renderedReplay, null);
  assert.equal(replayRequests.length, 2, 'retaining the selection must not fetch its detail twice');
  replayRequests[0].resolve(second);
  await pendingSecond;
  assert.equal(renderedReplay.window_id, 'second');

  const refreshBeforeSelection = replayFunctions.loadReplays(true);
  const pendingFirst = replayFunctions.selectReplay(1);
  replayRequests[2].resolve({windows: [second, first]});
  await refreshBeforeSelection;
  assert.equal(selectedReplayId, 'first', 'a selection made during list loading takes precedence');
  assert.equal(replayRequests.length, 4);
  const changeSelection = replayFunctions.selectReplay(0);
  replayRequests[3].resolve(first);
  await pendingFirst;
  assert.equal(renderedReplay, null, 'a late response cannot override a newer selection');
  replayRequests[4].resolve(second);
  await changeSelection;
  assert.equal(renderedReplay.window_id, 'second');

  const removedSelection = replayFunctions.selectReplay(0);
  const removeSelected = replayFunctions.loadReplays(true);
  replayRequests[6].resolve({windows: [first]});
  await tick();
  assert.equal(selectedReplayId, 'first');
  assert.equal(renderedReplay, null, 'removed detail clears while the replacement loads');
  replayRequests[5].resolve(second);
  await removedSelection;
  assert.equal(renderedReplay, null, 'removed record cannot return through a late detail response');
  replayRequests[7].resolve(first);
  await removeSelected;
  assert.equal(renderedReplay.window_id, 'first');

  const emptyHistory = replayFunctions.loadReplays(true);
  replayRequests[8].resolve({windows: []});
  await emptyHistory;
  assert.equal(selectedReplayId, undefined);
  assert.equal(renderedReplay, null, 'empty history clears the previously loaded detail');
  assert.equal(replayState.replayIndex, -1);
  replayState.replays = [second];
  const lateEmptyDetail = replayFunctions.selectReplay(0);
  const emptyPendingHistory = replayFunctions.loadReplays(true);
  replayRequests[10].resolve({windows: []});
  await emptyPendingHistory;
  replayRequests[9].resolve(second);
  await lateEmptyDetail;
  assert.equal(renderedReplay, null, 'late detail cannot repopulate empty history');
  assert.equal(selectedReplayId, undefined);
  console.log('PASS: serialized polling, replay selection races, timeouts, baseline freshness, and offline status');
})().catch(error => { console.error(error); process.exitCode = 1; });
