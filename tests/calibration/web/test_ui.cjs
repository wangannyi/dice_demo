// No DOM or hardware needed: verify completion never inherits a failed/stale attempt.
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const assert = require('node:assert/strict');
const source = readFileSync('calibration/web/static/app.js', 'utf8');
const functionSource = source.slice(source.indexOf('function completionFor('), source.indexOf('function renderCompletion('));
const base = () => ({active: false, recovery_required: false, busy: [],
  inspection_ready: true, actions: {plan: {ready: true}},
  state: {teaching_dataset: 'teach', plan: 'plan', registration: 'registration'},
  result: {quality_passed: true}, applied: true, table_ready: true, jobs: []});
function completed(state, action) {
  return !!runInNewContext(functionSource + '\ncompletionFor(action)', {state, action});
}
let checks = 0;
function expect(state, action, wanted) { assert.equal(completed(state, action), wanted, action); checks++; }
for (const action of ['first','plan','auto','register','restore','apply']) {
  const s = base();
  s.jobs = [{action, status: 'completed', started: 10}];
  expect(s, action, true);
  for (const status of ['running','stopping','failed','cancelled','interrupted']) {
    expect({...s, jobs: [{action, status, started: 11}, ...s.jobs]}, action, false);
  }
  expect({...s, active: true}, action, false);
  expect({...s, recovery_required: true}, action, false);
}
expect({...base(), busy: [{pid: 1}]}, 'inspect', false);
expect(base(), 'inspect', true);
expect({...base(), inspection_ready: false}, 'inspect', false);
expect(base(), 'window', true);
expect({...base(), actions: {plan: {ready: false}}}, 'window', false);
for (const [action, upstream] of [['plan','first'], ['auto','plan'], ['register','auto'], ['register','solve'], ['apply','restore']]) {
  expect({...base(), jobs: [{action: upstream, status: 'completed', started: 20},
    {action, status: 'completed', started: 10}]}, action, false);
}
expect({...base(), state: {plan: null}, jobs: [{action:'plan',status:'completed',started:10}]}, 'plan', false);
expect({...base(), table_ready: false, jobs: [{action:'apply',status:'completed',started:10}]}, 'apply', false);
expect({...base(), jobs: [{action:'table',status:'completed',started:12}, {action:'apply',status:'failed',started:10}]}, 'apply', true);
expect({...base(), result: {quality_passed:false}, jobs: [{action:'auto',status:'completed',started:10}]}, 'auto', true);
expect({...base(), state: {home_required: true}, jobs: [{action:'auto',status:'completed',started:10}]}, 'auto', false);
expect({...base(), state: {home_required: true, registration:'file'}, jobs: [{action:'register',status:'completed',started:10}]}, 'register', false);
console.log(`${checks} completion-state checks passed`);
const geometrySource = source.slice(source.indexOf('function liveRangeGeometry('), source.indexOf('function renderLiveRange('));
const visibility = {available:true,live_compatible:true,width:960,height:720,window_xyxy:[15,10,849,703]};
function geometry(v=visibility, frame={width:960,height:720}, w=960, h=720) {
  return runInNewContext(geometrySource + '\nliveRangeGeometry(v, frame, w, h)',{v,frame,w,h});
}
assert.equal(JSON.stringify(geometry()), JSON.stringify({x:15,y:10,width:834,height:693,viewBox:'0 0 960 720'}));
assert.equal(geometry({...visibility,live_compatible:false}),null);
assert.equal(geometry({...visibility,available:false}),null);
assert.equal(geometry(visibility,null),null);
assert.equal(geometry(visibility,{width:1280,height:720}),null);
assert.equal(geometry(visibility,{width:960,height:720},1280,720),null);
console.log('6 live overlay geometry checks passed');
