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
const progressSource = source.slice(source.indexOf('function jobProgress('), source.indexOf('function liveRangeGeometry('));
function progress(job) {
  return runInNewContext(progressSource + '\njobProgress(job)', {job});
}
for (const action of ['apply', 'table']) {
  for (const [script, message] of [
    ['calibration/apply_result.py', '正在应用标定结果…'],
    ['calibration/tools/table_capture.py', '正在拍摄桌面并拟合桌面平面…'],
    ['calibration/tools/register_home_table.py', '正在保存桌面配置…'],
  ]) {
    const p = progress({action, stage: {script}, progress: {total:20, accepted:0, message:'旧采样进度'}});
    assert.equal(p.counted, false);
    assert.equal(p.total, null);
    assert.equal(p.message, message);
  }
  assert.equal(progress({action}).counted, false);
}
const auto = {action:'auto',stage:{script:'calibration/auto_collect.py'}};
assert.equal(progress(auto).counted, false);
for (const total of [0, -1, NaN, Infinity, '20']) {
  assert.equal(progress({...auto,progress:{total}}).counted, false);
}
const sampling = progress({...auto,progress:{total:12,accepted:7,phase:'ACCEPTED'}});
assert.equal(sampling.counted, true);
assert.equal(sampling.total, 12);
assert.equal(sampling.count, 7);
for (const phase of ['HOME', 'HAND_HOME', 'HAND_COMMAND_SENT', 'RETURN_HOME', 'HOME_READY']) {
  assert.equal(progress({...auto,progress:{total:20,phase}}).counted, false);
}
for (const script of ['web_home_return','calibration/calibrate.py','calibration/tools/reprocess_dimensions.py']) {
  assert.equal(progress({...auto,stage:{script},progress:{total:20}}).counted, false);
}
for (const action of ['inspect','home','register','restore','plan','solve','first']) {
  assert.equal(progress({action,progress:{total:20}}).counted, false);
}
assert.equal(progress({action:'first',progress:{phase:'sampling',accepted:4}}).message, '已保存 4 组手眼样本');
console.log('Job progress checks passed: table stages, real auto totals, manual capture and HOME');
assert.equal(progress({...auto,progress:{phase:'MOVING to sample 7',total:20,accepted:6}}).message, '正在前往下一采样点…');
assert.equal(progress({action:'home',progress:{phase:'HAND_HOME',message:'internal details'}}).message, '正在将灵巧手归零…');
assert.equal(progress({action:'first',progress:{phase:'sampling',accepted:4,last_accepted:false,message:'Traceback'}}).message, '样本未保存，请调整姿态后重试。');
assert.equal(progress({action:'solve',progress:{phase:'internal debug value'}}).message, '正在处理，请稍候…');
const failureSource = source.slice(source.indexOf('function operatorFailure('), source.indexOf('function jobProgress('));
function failure(error) { return runInNewContext(failureSource + '\noperatorFailure(error)', {error}); }
assert.equal(failure('no fresh HOME Revo2 position feedback within 2s'), '未收到手指位置反馈，请检查灵巧手通信。');
assert.match(failure('未确认灵巧手六路归零；请检查手指反馈后重试回 HOME'), /灵巧手复位未完成/);
assert.match(failure('Too few ChArUco corners'), /标定板识别失败/);
assert.match(failure('RealSense disconnected'), /相机连接异常/);
assert.match(failure('Traceback: /tmp/private/session'), /操作记录与诊断/);
console.log('Concise operator message checks passed');
