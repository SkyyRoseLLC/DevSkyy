const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');
const source=fs.readFileSync(__dirname+'/../../wordpress-theme/skyyrose-flagship-2/assets/js/mascot.js','utf8');
for(const moved of [false,true])test(`delayed guide exit ${moved?'does not steal changed focus':'returns guide focus'}`,()=>{
 const trigger={},other={},body={};const document={activeElement:trigger,body};let done,focused=false;
 const context={document,mascotEl:{contains:e=>e===trigger},recallBtn:{style:{},setAttribute(){},focus(){focused=true;}},markDismissed(){},walkOff:callback=>{done=callback;}};
 vm.createContext(context);vm.runInContext(source.slice(source.indexOf('\tfunction minimize('),source.indexOf('\tfunction recall('))+'\nminimize(true);',context);
 if(moved)document.activeElement=other;done();assert.equal(focused,!moved);
});
