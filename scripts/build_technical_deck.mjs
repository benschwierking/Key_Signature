// Build editable PowerPoint evidence from measured experiment outputs.
// Requires the Codex presentation runtime. Paths are supplied through env.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [inputPath, workspaceDir, finalPath] = process.argv.slice(2);
if (!inputPath || !workspaceDir || !finalPath) throw new Error('Usage: build_technical_deck.mjs DATA_JSON BUILD_DIR FINAL_PPTX');
const modules = process.env.KEYCLF_ARTIFACT_MODULES;
const skill = process.env.KEYCLF_PRESENTATION_SKILL;
const python = process.env.KEYCLF_RUNTIME_PYTHON;
const buildDir=path.join(workspaceDir,'.deck-build');
if (!modules || !skill || !python) throw new Error('Presentation runtime paths are required in KEYCLF_* environment variables');
process.env.RUNTIME_NODE_MODULES ??= modules;
const require = createRequire(path.join(modules, '__keyclf_runtime__.cjs'));
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const { applyPresentationChartFont, resolvePresentationFont, finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const font = resolvePresentationFont({fontFamily:'Arial'});
const d = JSON.parse(await fs.readFile(inputPath,'utf8'));
const exp=d.research, selected=exp.selected, test=exp.final.test, fit=exp.final.train_resubstitution;
const baseline=d.baseline_test, p=d.projection;
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#17324D',blue:'#2563EB',teal:'#0D9488',orange:'#D97706',gray:'#64748B',light:'#E2E8F0',ink:'#172B40'};
const palette=['#2563EB','#DC2626','#059669','#7C3AED','#D97706','#0891B2','#DB2777','#65A30D','#4F46E5','#9333EA','#0F766E','#B45309'];
const pct=x=>(100*x).toFixed(1)+'%';
const keyNames=Object.keys(exp.audit.class_counts);
const pitch=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const chartOwners=[], tableOwners=[];
function text(s,value,x,y,w,h,size=26,color=C.ink,bold=false){
  const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  o.text=value;o.text.style={typeface:font,fontSize:size,color,bold,wrap:'square',autoFit:'none',insets:0};return o;
}
function slide(title,note=''){
  const s=P.slides.add();s.background.fill='#FFFFFF';
  text(s,title,56,38,1168,74,44,C.navy,true);
  text(s,String(P.slides.items.length).padStart(2,'0'),1190,676,40,22,16,C.gray);
  s.speakerNotes.text=note;return s;
}
function note(s,value){s.speakerNotes.text=String(s.speakerNotes.text||'')+'\n'+value;}
function table(s,values,widths,{x=56,y=150,w=1168,h=420,size=24}={}){
  tableOwners.push(P.slides.items.length);
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({fill:C.light,width:1,style:'solid'});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
    const cell=t.getCell(r,c);cell.text.style={typeface:font,fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0,insets:12};
    cell.fill=r===0?C.navy:(r%2?'#F5F8FC':'#FFFFFF');
  }
  return t;
}
function chart(s,type,config){
  chartOwners.push(P.slides.items.length);
  const c=s.charts.add(type,{position:{left:56,top:150,width:1168,height:430},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},
    xAxis:{textStyle:{fontSize:18,fill:C.gray},line:{fill:C.light,width:1}},
    yAxis:{textStyle:{fontSize:18,fill:C.gray},majorGridlines:{fill:C.light,width:1}},
    legend:{position:'bottom',textStyle:{fontSize:18}},...config});
  applyPresentationChartFont(c,{fontFamily:font});return c;
}
function bar(s,categories,series,{percent=true,...extra}={}){
  return chart(s,'bar',{categories,series,hasLegend:series.length>1,barOptions:{direction:'column',grouping:'clustered',gapWidth:60},
    yAxis:{min:0,...(percent?{max:1,numberFormatCode:'0%'}:{}),textStyle:{fontSize:18,fill:C.gray},majorGridlines:{fill:C.light,width:1}},...extra});
}
function scatter(s,points,xField,yField,groupFn,labels,colors,extra={}){
  const series=labels.map((label,i)=>{const a=points.filter(q=>groupFn(q)===label);return{name:label,xValues:a.map(q=>q[xField]),values:a.map(q=>q[yField]),fill:colors[i%colors.length],line:{fill:'none',width:0},marker:{symbol:'circle',size:5,fill:colors[i%colors.length]}};});
  const xs=points.map(q=>q[xField]),ys=points.map(q=>q[yField]);
  const xmin=Math.floor(Math.min(...xs)-0.5),xmax=Math.ceil(Math.max(...xs)+0.5);
  const ymin=Math.floor(Math.min(...ys)-0.5),ymax=Math.ceil(Math.max(...ys)+0.5);
  return chart(s,'scatter',{series,scatterOptions:{style:'marker'},hasLegend:true,...extra,
    xAxis:{...extra.xAxis,min:xmin,max:xmax,crossesAt:ymin,tickLabelPosition:'low'},
    yAxis:{...extra.yAxis,min:ymin,max:ymax,crossesAt:xmin,tickLabelPosition:'low'}});
}

// 1: Minimal title.
{
 const s=P.slides.add();s.background.fill=C.navy;
 text(s,'Key signature\nclassification',64,155,1120,190,68,'#FFFFFF',true);
 text(s,'Training, musical features and evaluation',68,405,1100,60,30,'#D9E6F4');
 text(s,'Ben Schwierking\n30 October 2026',68,570,900,80,25,'#D9E6F4');
 s.speakerNotes.text='Measured technical review of the v1 classifier and the opt-in v2 research experiments. Source repository: https://github.com/benschwierking/Key_Signature. Research outputs remain local pending data redistribution review.';
}
// 2: Current method.
{
 const s=slide('The original model receives 36 chroma features','Source: keyclf/features.py and specs/key-signature-classifier-spec.md. Chroma folds octaves into twelve pitch classes. The edge window spans approximately 20*512/22050 seconds.');
 table(s,[['Feature block','Definition','Dimensions'],['First pitch class','Dominant pitch over the first 20 trimmed frames','12'],['Last pitch class','Dominant pitch over the last 20 trimmed frames','12'],['Chroma histogram','Normalized pitch-class energy across the preview','12']],[260,740,168],{h:325});
 text(s,'The edge features describe dominant pitch content, rather than transcribed first and last notes.',56,520,1150,85,28,C.navy);
}
// 3: Data and splits.
{
 const s=slide('Dataset and fixed recording splits','Sources: GiantSteps original annotations, research audit.json. The original split IDs remain fixed. All augmentation happens after assigning original recordings to folds.');
 const n=exp.audit.n, c=exp.audit.split_counts;
 table(s,[['Partition','Recordings','Share','Purpose'],['Training',String(c.train),pct(c.train/n),'Original model fitting'],['Validation',String(c.val),pct(c.val/n),'Original model selection'],['Test',String(c.test),pct(c.test/n),'Final fixed benchmark'],['Final refit',String(c.train+c.val),pct((c.train+c.val)/n),'Fit the selected configuration']],[245,185,175,563],{h:345});
 text(s,'Research selection uses three development folds across the 513 training and validation recordings. Percentages rounded.',56,545,1150,86,27,C.navy);
 note(s,'Reference: https://github.com/GiantSteps/giantsteps-key-dataset. All 604 excerpts are from electronic dance music. New research reuses the existing test set as an audit benchmark because its errors have already been examined.');
}
// 4: Imbalance.
{
 const s=slide('Minor keys dominate the training source','Source: research audit.json class counts across all 604 recordings.');
 bar(s,pitch,[{name:'Major',values:pitch.map(k=>exp.audit.class_counts[k+' major']),fill:C.orange},{name:'Minor',values:pitch.map(k=>exp.audit.class_counts[k+' minor']),fill:C.blue}],{percent:false});
 text(s,`${exp.audit.mode_counts.minor} minor recordings and ${exp.audit.mode_counts.major} major recordings. D-flat major has one recording.`,56,610,1140,68,26,C.navy);
}
// 5: Original training.
{
 const s=slide('Original training selected a random forest','Source: keyclf/train.py and reports/training_metrics.json. Logistic regression uses lbfgs with max_iter 5000. Candidate selection used exact validation accuracy, then top-three accuracy, then lower log loss. The forest refits on train plus validation.');
 table(s,[['Algorithm','Candidates','Settings'],['Multinomial logistic regression','8','C = 0.01, 0.1, 1, 10 with optional class weights'],['Random forest','6','500 trees, leaf sizes 1, 2, 4 and two feature subsets']],[345,170,653],{h:255});
 text(s,'Selected forest: 500 trees, leaf size 1, max_features 0.5, balanced_subsample, seed 42.',56,470,1150,90,29,C.navy);
 text(s,'Original validation accuracy: 63.7%. Original held-out accuracy: 46.2%.',56,590,1150,45,26,C.gray);
}
// 6: Research protocol.
{
 const s=slide('Research protocol and leakage controls','Sources: keyclf/experiments.py, fold_ids.json and experiment_results.json. Mode stratification handles the single-recording tonic class. Artist metadata is unavailable, so grouping guarantees apply to recording IDs, not artist identities.');
 text(s,'Three folds across development recordings',56,155,1160,55,34,C.blue,true);
 text(s,'Each fold fits scaling and model parameters on its training recordings. Validation recordings remain original and unaugmented.',56,225,1140,90,28);
 text(s,'Training augmentation rotates every pitch-class block through 12 semitones and updates the tonic label. Major or minor mode stays unchanged.',56,345,1140,100,28);
 text(s,'Selection order: exact accuracy, top-three accuracy, lower log loss. One final fit precedes the fixed test evaluation.',56,485,1140,105,28,C.navy);
 text(s,'No test rows participate in feature selection or cross-validation fitting.',56,615,1140,40,24,C.gray);
}
// 7: Musical extensions.
{
 const s=slide('New features retain more musical context','Source: keyclf/research_features.py. The full research cache has 15 blocks of 12 pitch classes, for 180 features. References: https://essentia.upf.edu/reference/streaming_KeyExtractor.html and https://music21.org/music21docs/_modules/music21/analysis/discrete.html.');
 table(s,[['Feature family','What the extraction measures'],['Harmonic content','Pitch-class content after harmonic/percussive separation'],['Bass content','Separate chroma summary from the lowest three octaves'],['Beat and accent','Equal-beat summaries and onset-accent weighting'],['Pitch persistence','Sustained salience across neighboring frames'],['Temporal context','Four successive sections and harmonic edge profiles']],[335,833],{h:395,size:24});
 text(s,'Beat accents do not identify bar downbeats. Persistence does not transcribe individual note durations.',56,590,1140,72,25,C.orange);
}
// 8: CV comparison.
{
 const s=slide('Feature experiments on development folds','Source: experiment_results.json. Bars represent pooled out-of-fold results over the same 513 development tracks. Each feature-family bar uses its highest-ranked algorithm, so this chart compares selected configurations rather than isolating only the features.');
 const groups=[...new Set(exp.candidates.map(r=>r.config.group))];
 const best=groups.map(g=>exp.candidates.filter(r=>r.config.group===g).sort((a,b)=>b.development_cv.accuracy-a.development_cv.accuracy)[0]);
 const labels={v1:'V1 + transpose',histogram:'Histogram',harmonic_bass:'Harmonic + bass',persistence:'Persistence',beat_accent:'Beat accent',temporal:'Sections',all_features:'All features'};
 bar(s,best.map(r=>labels[r.config.group]||r.config.group),[{name:'Exact accuracy',values:best.map(r=>r.development_cv.accuracy),fill:C.blue},{name:'Macro recall',values:best.map(r=>r.development_cv.macro_recall_present),fill:C.teal}]);
 text(s,`Selected: ${selected.config.group.replaceAll('_',' ')} with ${selected.config.algorithm==='forest'?'random forest':'scaled logistic regression'}. CV exact accuracy ${pct(selected.development_cv.accuracy)}.`,56,610,1140,65,25,C.navy);
}
// 9: Algorithm settings.
{
 const s=slide('Selected research model','Sources: final_test.json, experiment_results.json and research_model.joblib. Feature names and training split IDs are stored with the artifact.');
 const cfg=selected.config;
 table(s,[['Setting','Value'],['Algorithm',cfg.algorithm==='forest'?'Random forest classifier':'StandardScaler + multinomial logistic regression'],['Feature family',cfg.group.replaceAll('_',' ')],['Input dimensions',String(exp.final.feature_names.length)],['Parameters',cfg.algorithm==='forest'?`500 trees, min_samples_leaf ${cfg.leaf}, max_features ${cfg.max_features}`:`C ${cfg.C}, lbfgs solver`],['Class weights',cfg.algorithm==='forest'?'balanced_subsample':'balanced'],['Training examples',`${exp.final.fit_recordings} original recordings, ${exp.final.augmented_training_rows} rows after augmentation`],['Random seed','42']],[340,828],{h:435,size:24});
}
// 10: Train versus test.
{
 const s=slide('Training fit and held-out performance','Sources: original v1 refit artifact and final_test.json. Training figures are resubstitution on 513 original development recordings after fitting. CV predictions are out-of-fold. The fixed test set contains 91 tracks. Training accuracy does not estimate generalization.');
 table(s,[['Metric','Original V1','Research model'],['Training accuracy',pct(d.baseline_fit.accuracy),pct(fit.accuracy)],['Training macro recall',pct(d.baseline_fit.macro_recall_present),pct(fit.macro_recall_present)],['Development CV accuracy',pct(d.baseline_cv.accuracy),pct(selected.development_cv.accuracy)],['Test exact accuracy',pct(baseline.accuracy),pct(test.accuracy)],['Test top-three accuracy',pct(baseline.top3_accuracy),pct(test.top3_accuracy)],['Test macro recall',pct(baseline.macro_recall_present),pct(test.macro_recall_present)],['Test weighted recall',pct(baseline.weighted_recall),pct(test.weighted_recall)],['Test log loss',baseline.log_loss.toFixed(3),test.log_loss.toFixed(3)]],[530,319,319],{h:465,size:23});
 text(s,'Targets remain 80% exact and 95% top-three. Test macro recall averages the 23 represented keys.',56,644,1140,42,21,C.gray);
 note(s,'This table recomputes both log losses with sklearn default float64 clipping. The original acceptance report uses EPS=1e-15, yielding 2.337 rather than 2.353 for the original model. This numeric convention does not change accuracy, recall, ranking, or predictions.');
}
// 11: Recall meaning.
{
 const s=slide('Recall by key and mode','Sources: classification_report fields in final_test.json and original model evaluation. In single-label multiclass classification, weighted recall equals accuracy. Macro recall gives every represented class equal weight.');
 text(s,'Recall for a key = correct predictions of that key / actual recordings in that key',56,150,1150,85,32,C.blue,true);
 table(s,[['Test measure','Tracks','Original V1','Research model'],['Major exact recall',String(test.major_n),pct(baseline.major_exact_recall),pct(test.major_exact_recall)],['Minor exact recall',String(test.minor_n),pct(baseline.minor_exact_recall),pct(test.minor_exact_recall)],['Mode accuracy','91',pct(baseline.mode_accuracy),pct(test.mode_accuracy)]],[300,140,364,364],{y:280,h:255,size:24});
 text(s,'Mode accuracy checks major versus minor only. It does not require the correct tonic.',56,585,1140,70,27,C.gray);
}
// 12 and 13: Per-class recall.
for(const mode of ['major','minor']){
 const s=slide(`Test recall by ${mode} key`,'Source: class-specific recall and support in classification_report. Zero-support keys have no estimable test recall and are marked with an asterisk. No label or split was changed after observing errors.');
 const names=pitch.map(k=>k+' '+mode);
 const present=names.map(k=>test.classification_report[k].support);
 bar(s,names.map((k,i)=>pitch[i]+(present[i]?'':'*')),[{name:'Original V1',values:names.map(k=>baseline.classification_report[k].recall),fill:C.gray},{name:'Research',values:names.map(k=>test.classification_report[k].recall),fill:C.blue}]);
 text(s,'Test support by tonic: '+present.join(', ')+(mode==='major'?'.  * No D-flat-major test example.':'.'),56,610,1140,60,24,C.gray);
}
// 14: PCA.
{
 const s=slide('Feature-space projection by major/minor mode','Source: projection_points.json and projection_metrics.json. StandardScaler and PCA fit development features only. Test points are transformed afterward. PCA is a visualization, not the classifier.');
 scatter(s,d.points,'pca1','pca2',q=>`${q.mode} ${q.split==='test'?'test':'development'}`,['major development','minor development','major test','minor test'],[C.orange,C.blue,'#7C2D12',C.teal],{xAxis:{title:'Principal component 1',textStyle:{fontSize:18}},yAxis:{title:'Principal component 2',textStyle:{fontSize:18}}});
 text(s,`The first two components explain ${pct(p.pca_explained_variance.reduce((a,b)=>a+b,0))} of standardized feature variance.`,56,610,1140,65,25,C.navy);
}
// 15: t-SNE.
{
 const s=slide('Local neighborhoods by tonic','Source: projection_points.json. t-SNE uses development recordings only, perplexity 30, PCA initialization, seed 42 and 1000 iterations. Axes are arbitrary. Distances between isolated groups are not calibrated measures of musical similarity.');
 scatter(s,d.points.filter(q=>q.split!=='test'),'tsne1','tsne2',q=>pitch[q.tonic],pitch,palette,{xAxis:{title:'t-SNE dimension 1',textStyle:{fontSize:18}},yAxis:{title:'t-SNE dimension 2',textStyle:{fontSize:18}}});
 text(s,'Colors show known tonic labels. This plot explores feature neighborhoods and does not establish test accuracy.',56,610,1140,65,25,C.gray);
}
// 16: Actual clusters.
{
 const s=slide('Exploratory K-means clusters','Source: projection_metrics.json. K-means fits 24 clusters to standardized development features, n_init 20, seed 42. The scatter uses the same PCA projection. Cluster IDs are arbitrary and do not correspond one-to-one with musical keys.');
 const labels=Array.from({length:24},(_,i)=>String(i));
 scatter(s,d.points.filter(q=>q.split!=='test'),'pca1','pca2',q=>String(q.cluster),labels,[...palette,...palette.map((c,i)=>['#93C5FD','#FCA5A5','#6EE7B7','#C4B5FD','#FCD34D','#67E8F9','#F9A8D4','#BEF264','#A5B4FC','#D8B4FE','#5EEAD4','#FDE68A'][i])],{hasLegend:false,position:{left:56,top:150,width:830,height:450},xAxis:{title:'Principal component 1',textStyle:{fontSize:18}},yAxis:{title:'Principal component 2',textStyle:{fontSize:18}}});
 text(s,`24 clusters\n\nSilhouette\n${p.kmeans_silhouette.toFixed(3)}\n\nAdjusted Rand\n${p.kmeans_adjusted_rand.toFixed(3)}`,925,180,290,350,30,C.navy);
 text(s,'The prediction algorithm is supervised classification. K-means serves only as an exploratory diagnostic.',56,625,1140,40,24,C.gray);
}
// 17: Learning curve.
{
 const s=slide('Learning curve on development recordings','Source: learning_curve.json. Each point averages three folds. Training subsets are stratified by mode. The x-axis counts independent original recordings before transposition. The same development validation folds remain fixed across budgets. No held-out test predictions are used.');
 chart(s,'line',{categories:d.learning_curve.map(q=>String(Math.round(q.mean_recordings))),series:[{name:'Mean CV accuracy',values:d.learning_curve.map(q=>q.mean_accuracy),line:{fill:C.blue,width:3},marker:{symbol:'circle',size:8}}],hasLegend:false,
  xAxis:{title:'Original training recordings per fold',textStyle:{fontSize:18}},yAxis:{title:'Validation exact accuracy',min:0,max:1,numberFormatCode:'0%',textStyle:{fontSize:18},majorGridlines:{fill:C.light,width:1}}});
 text(s,'This experiment measures gains within the current dataset. More independent major-key recordings remain the priority for collection.',56,610,1140,65,25,C.navy);
}
// 18: Limits and next work.
{
 const s=slide('Remaining technical limits','Sources: experiment_results.json, extraction diagnostics and implementation status. No additional external dataset was acquired for this experiment. Novel feature extraction has not yet received the full original C1-C7 acceptance certification.');
 text(s,`Measured test exact accuracy: ${pct(test.accuracy)}. Top-three accuracy: ${pct(test.top3_accuracy)}.`,56,155,1160,60,33,C.blue,true);
 text(s,'The current dataset has few major-key examples and uneven support across tonics. Additional data should contain independently verified major keys.',56,245,1140,105,28);
 text(s,'The tested extensions estimate beat accents and persistent pitch. Exact note onsets, offsets and bar downbeats remain future experiments.',56,380,1140,100,28);
 text(s,d.research_latency?`The selected histogram model and its extraction path took ${d.research_latency.max_s.toFixed(2)} seconds maximum across three 120-second synthetic MP3 runs. A fresh blind benchmark remains necessary.`:'The original test errors have been examined. A later release needs a fresh blind benchmark and a measured latency gate for the new extractor.',56,515,1140,105,28,C.navy);
 note(s,'Research references: https://archives.ismir.net/ismir2021/paper/000004.pdf. Pitch shifting can improve transposition coverage, but it preserves mode and does not replace independent major-key examples.');
}
// 19: Concrete additional-data pool.
if(d.additional_data.mode_counts){
 const a=d.additional_data;
 const s=slide('Additional data candidates','Source: additional_data_audit.json and the current GiantSteps-MTG annotations. https://github.com/GiantSteps/giantsteps-mtg-key-dataset. Counts refer to single-key annotations after parsing and exact ID/MD5 overlap checks. Near duplicates, artist relationships, audio availability and usage terms still need review.');
 table(s,[['Metadata audit','Count'],['Single-key candidate recordings',String(a.independent_candidates)],['Major-key candidates',String(a.mode_counts.major)],['Minor-key candidates',String(a.mode_counts.minor)],['Unknown or ambiguous entries',String(a.rejected_annotations.length)],['Exact ID or MD5 overlaps',String(a.exact_id_or_md5_overlap)]],[740,428],{h:400});
 text(s,'These recordings have not entered the current model. The next collection stage prioritizes verified major keys.',56,600,1140,65,26,C.navy);
}

await fs.mkdir(buildDir,{recursive:true});
const previews=path.join(buildDir,'previews');await fs.mkdir(previews,{recursive:true});
for(let i=0;i<P.slides.items.length;i++){
 const png=await P.export({slide:P.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const montage=await P.export({format:'webp',montage:true});
await fs.writeFile(path.join(buildDir,'montage.webp'),new Uint8Array(await montage.arrayBuffer()));
const candidatePath=path.join(buildDir,'draft.pptx');
await(await PresentationFile.exportPptx(P)).save(candidatePath);
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,
 pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(i=>['--require-native-table-slide',String(i)])],
 requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,
 materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'powerpoint',
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(buildDir,'validation.json')});
console.log(JSON.stringify({slides:P.slides.items.length,output:finalPath,result}));
