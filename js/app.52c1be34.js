(function(e){function a(a){for(var n,i,d=a[0],s=a[1],l=a[2],m=0,x=[];m<d.length;m++)i=d[m],Object.prototype.hasOwnProperty.call(o,i)&&o[i]&&x.push(o[i][0]),o[i]=0;for(n in s)Object.prototype.hasOwnProperty.call(s,n)&&(e[n]=s[n]);c&&c(a);while(x.length)x.shift()();return r.push.apply(r,l||[]),t()}function t(){for(var e,a=0;a<r.length;a++){for(var t=r[a],n=!0,d=1;d<t.length;d++){var s=t[d];0!==o[s]&&(n=!1)}n&&(r.splice(a--,1),e=i(i.s=t[0]))}return e}var n={},o={app:0},r=[];function i(a){if(n[a])return n[a].exports;var t=n[a]={i:a,l:!1,exports:{}};return e[a].call(t.exports,t,t.exports,i),t.l=!0,t.exports}i.m=e,i.c=n,i.d=function(e,a,t){i.o(e,a)||Object.defineProperty(e,a,{enumerable:!0,get:t})},i.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.t=function(e,a){if(1&a&&(e=i(e)),8&a)return e;if(4&a&&"object"===typeof e&&e&&e.__esModule)return e;var t=Object.create(null);if(i.r(t),Object.defineProperty(t,"default",{enumerable:!0,value:e}),2&a&&"string"!=typeof e)for(var n in e)i.d(t,n,function(a){return e[a]}.bind(null,n));return t},i.n=function(e){var a=e&&e.__esModule?function(){return e["default"]}:function(){return e};return i.d(a,"a",a),a},i.o=function(e,a){return Object.prototype.hasOwnProperty.call(e,a)},i.p="/reviews-sentiment/";var d=window["webpackJsonp"]=window["webpackJsonp"]||[],s=d.push.bind(d);d.push=a,d=d.slice();for(var l=0;l<d.length;l++)a(d[l]);var c=s;r.push([0,"chunk-vendors"]),t()})({0:function(e,a,t){e.exports=t("56d7")},"034f":function(e,a,t){"use strict";var n=t("8a23"),o=t.n(n);o.a},"56d7":function(e,a,t){"use strict";t.r(a);var n=t("2b0e"),o=function(){var e=this,a=e.$createElement,t=e._self._c||a;return t("Demo")},r=[],i=function(){var e=this,a=e.$createElement,t=e._self._c||a;return t("v-app",[t("v-content",[t("v-container",[t("v-layout",{attrs:{"text-center":"",wrap:""}},[t("v-flex",{attrs:{xs12:""}},[t("h1",{staticClass:"title1"},[e._v(" Amazon Reviews Demo ")]),t("p",{staticClass:"subheading font-weight-regular"},[e._v(" Data Analytics project, Coppola, Palazzi, Vivace - January 2020 ")])]),t("v-tabs",{attrs:{grow:!0,right:"right","align-with-title":"","background-color":"transparent"}},[t("v-tab",{staticStyle:{"font-size":"1.2rem"},on:{click:function(a){e.toggledexploration=!e.toggledexploration,e.toggledlda=!1,e.toggledsent=!1}}},[e._v(" EXPLORATION")]),t("v-tab",{staticStyle:{"font-size":"1.2rem"},on:{click:function(a){e.toggledlda=!e.toggledlda,e.toggledsent=!1,e.toggledexploration=!1}}},[e._v(" LDA ")]),t("v-tab",{staticStyle:{"font-size":"1.2rem"},on:{click:function(a){e.toggledsent=!e.toggledsent,e.toggledlda=!1,e.toggledexploration=!1}}},[e._v(" SENTIMENT ANALYSIS ")])],1),e.toggledexploration?t("v-flex",{attrs:{lg12:"",xs12:""}},[t("br"),t("br"),t("v-row",{attrs:{justify:"center"}},[t("v-col",{attrs:{cols:"8",sm:"8"}},[t("v-select",{attrs:{items:e.plots,label:"Select plot to show",solo:""},model:{value:e.selectedPlot,callback:function(a){e.selectedPlot=a},expression:"selectedPlot"}})],1)],1),t("v-row",{attrs:{justify:"center"}},[t("v-col",{attrs:{cols:"12",sm:"12"}},[t("h3"),t("br"),1==e.selectedPlot?[t("h3",[e._v(" Published reviews timeseries ")]),t("apexchart",{attrs:{width:"100%",height:"460px",options:e.timeseriesPlot,series:e.timeseries}}),t("h3",[e._v(" Reviews per week day ")]),t("apexchart",{attrs:{width:"100%",height:"460px",options:e.weekdayPlot,series:e.weekdaySeries}}),t("h3",[e._v(" Reviews per month ")]),t("apexchart",{attrs:{width:"100%",height:"460px",options:e.monthPlot,series:e.monthSeries}})]:e._e(),0==e.selectedPlot?[t("h3",[e._v(" Verified VS Unverified average score ")]),e._v(" Click on a bar to view the detailed distribution of reviews for that product "),t("apexchart",{attrs:{width:"100%",height:"460px",options:e.chartOptions,series:e.series,type:"bar"},on:{dataPointSelection:e.clickedPlot}})]:e._e(),t("br"),null!=e.selectedProduct&&0==e.selectedPlot?[t("h3",[e._v(" Review count per rating for the product "+e._s(this.chartOptions.xaxis.categories[e.selectedProduct])+" ")]),t("apexchart",{attrs:{type:"bar",height:"250",options:e.reviewCountPlot,series:e.count[e.selectedProduct]}})]:e._e()],2)],1)],1):e._e(),e.toggledsent?t("v-flex",{attrs:{lg12:"",xs12:""}},[t("v-row",{attrs:{justify:"center"}},[t("v-col",{attrs:{cols:"6",sm:"6"}},[t("br"),e._v(" Write a custom review to see the evaluated sentiment: "),t("v-text-field",{attrs:{label:"Custom Review",outlined:"",clearable:"",counter:"",hint:"English only!"},on:{change:e.compute},model:{value:e.formText,callback:function(a){e.formText=a},expression:"formText"}})],1)],1),t("v-row",{attrs:{justify:"center"}},[e.errormsg?t("center",[t("p",{staticClass:"red"},[e._v(" "+e._s(e.errormsg)+" "),t("br")]),e._v(" Did you check the backend is running correctly? ")]):t("center",[e._v(" "+e._s((100*e.value.toFixed(5)).toFixed(2))+"% ")])],1),t("br"),t("v-btn",{on:{click:e.compute}},[e._v("compute ")])],1):e._e(),e.toggledlda?t("v-flex",{attrs:{xs12:""}},[t("v-container",{attrs:{fluid:""}},[t("v-row",{attrs:{dense:""}},e._l(e.lda,(function(a){return t("v-col",{key:a.nam,attrs:{cols:"4"}},[t("v-card",{key:a,staticClass:"mx-auto",attrs:{"max-width":"344"}},[t("v-img",{attrs:{src:a.code+".jpg",height:"200px"}}),t("v-card-title",[e._v(" "+e._s(a.name)+" ")]),t("v-card-subtitle",[e._v(" "+e._s(a.description)+" ")]),t("v-card-actions",[t("v-btn",{attrs:{href:"lda_"+a.code+".html",target:"_blank",text:""}},[e._v(" pyLDAvis"),t("v-icon",{attrs:{right:"",dark:""}},[e._v("mdi-open-in-new")])],1)],1)],1)],1)})),1)],1)],1):e._e()],1)],1)],1)],1)},d=[],s={name:"Demo",methods:{clickedPlot:function(e,a,t){console.log("Selected product:",this.chartOptions.xaxis.categories[t.dataPointIndex]),this.selectedProduct=t.dataPointIndex},compute:function(){this.errormsg=null;let e=this;this.$axios.get("http://localhost:5000/",{params:{text:this.formText}}).then((function(a){e.value=a.data.positive})).catch((function(a){e.errormsg=a+" on "+e.apiEndpoint}))}},watch:{formText:function(e){this.test="A"+e}},data:()=>({apiEndpoint:"http://localhost:5000/",errormsg:null,formText:"",test:"aa",toggledlda:!1,toggledsent:!1,value:0,toggledexploration:!0,selectedPlot:0,selectedProduct:null,plots:[{value:0,text:"Verified VS Unverified study"},{value:1,text:"Time series"}],timeseries:[{name:"Unverified",data:[{x:13595904e5,y:694},{x:13620096e5,y:548},{x:1364688e6,y:537},{x:136728e7,y:501},{x:13699584e5,y:607},{x:13725504e5,y:587},{x:13752288e5,y:663},{x:13779072e5,y:583},{x:13804992e5,y:532},{x:13831776e5,y:665},{x:13857696e5,y:605},{x:1388448e6,y:732},{x:13911264e5,y:795},{x:13935456e5,y:682},{x:1396224e6,y:824},{x:1398816e6,y:868},{x:14014944e5,y:985},{x:14040864e5,y:1847},{x:14067648e5,y:4862},{x:14094432e5,y:5429},{x:14120352e5,y:5189},{x:14147136e5,y:5991},{x:14173056e5,y:3784},{x:1419984e6,y:2402},{x:14226624e5,y:2805},{x:14250816e5,y:2738},{x:142776e7,y:3529},{x:1430352e6,y:3527},{x:14330304e5,y:3331},{x:14356224e5,y:2781},{x:14383008e5,y:2593},{x:14409792e5,y:2829},{x:14435712e5,y:3360},{x:14462496e5,y:3333},{x:14488416e5,y:4304},{x:145152e7,y:4071},{x:14541984e5,y:4149},{x:1456704e6,y:3320},{x:14593824e5,y:4051},{x:14619744e5,y:4759},{x:14646528e5,y:4484},{x:14672448e5,y:6143},{x:14699232e5,y:4830},{x:14726016e5,y:3115},{x:14751936e5,y:2665},{x:1477872e6,y:1869},{x:1480464e6,y:1260},{x:14831424e5,y:1396},{x:14858208e5,y:1132},{x:148824e7,y:723},{x:14909184e5,y:800},{x:14935104e5,y:746},{x:14961888e5,y:666},{x:14987808e5,y:666},{x:15014592e5,y:833},{x:15041376e5,y:679},{x:15067296e5,y:500},{x:1509408e6,y:459},{x:1512e9,y:407},{x:15146784e5,y:403}]},{name:"Total",data:[{x:13595904e5,y:7389},{x:13620096e5,y:6121},{x:1364688e6,y:6232},{x:136728e7,y:5897},{x:13699584e5,y:6590},{x:13725504e5,y:6730},{x:13752288e5,y:7585},{x:13779072e5,y:7784},{x:13804992e5,y:6660},{x:13831776e5,y:7830},{x:13857696e5,y:7970},{x:1388448e6,y:9832},{x:13911264e5,y:10697},{x:13935456e5,y:8668},{x:1396224e6,y:9675},{x:1398816e6,y:9086},{x:14014944e5,y:9135},{x:14040864e5,y:9717},{x:14067648e5,y:17970},{x:14094432e5,y:19101},{x:14120352e5,y:18890},{x:14147136e5,y:22104},{x:14173056e5,y:22577},{x:1419984e6,y:27335},{x:14226624e5,y:28805},{x:14250816e5,y:28347},{x:142776e7,y:28717},{x:1430352e6,y:25657},{x:14330304e5,y:24539},{x:14356224e5,y:24519},{x:14383008e5,y:25943},{x:14409792e5,y:26416},{x:14435712e5,y:26011},{x:14462496e5,y:28206},{x:14488416e5,y:27989},{x:145152e7,y:27912},{x:14541984e5,y:30385},{x:1456704e6,y:26183},{x:14593824e5,y:29246},{x:14619744e5,y:28392},{x:14646528e5,y:28183},{x:14672448e5,y:28803},{x:14699232e5,y:28538},{x:14726016e5,y:27613},{x:14751936e5,y:24331},{x:1477872e6,y:20732},{x:1480464e6,y:18169},{x:14831424e5,y:20972},{x:14858208e5,y:20163},{x:148824e7,y:14324},{x:14909184e5,y:15718},{x:14935104e5,y:12842},{x:14961888e5,y:11181},{x:14987808e5,y:10209},{x:15014592e5,y:10101},{x:15041376e5,y:9644},{x:15067296e5,y:8017},{x:1509408e6,y:7343},{x:1512e9,y:7196},{x:15146784e5,y:6432}]}],timeseriesPlot:{fill:{type:"gradient",gradient:{shadeIntensity:.5,inverseColors:!1,opacityFrom:1,opacityTo:.9,stops:[0,90,100]}},legend:{fontSize:"18px"},chart:{toolbar:{show:!0},height:350},xaxis:{type:"datetime"}},weekdayPlot:{dataLabels:{enabled:!0},chart:{type:"bar",height:350,stacked:!0,toolbar:{show:!1}},plotOptions:{bar:{horizontal:!1}},xaxis:{categories:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]},yaxis:{labels:{formatter:function(e){return e/1e3+"K"}}}},monthPlot:{dataLabels:{enabled:!0},chart:{type:"bar",height:350,stacked:!0,toolbar:{show:!1}},plotOptions:{bar:{horizontal:!1}},xaxis:{categories:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]},yaxis:{labels:{formatter:function(e){return e/1e3+"K"}}}},weekdaySeries:[{name:"1",data:[12638,12838,12481,12010,11258,10081,10200]},{name:"2",data:[8944,8997,8757,8461,8025,6855,7127]},{name:"3",data:[15531,15521,15275,14627,13721,11695,11843]},{name:"4",data:[29982,29097,28286,27227,25107,22009,22638]},{name:"5",data:[114477,110927,108314,104506,96903,84637,86659]}],monthSeries:[{name:"1",data:[7308,6308,6868,6610,6357,6355,7339,7120,6647,6705,6617,7272]},{name:"2",data:[5407,4384,4760,4569,4479,4472,5009,5111,4417,4672,4559,5327]},{name:"3",data:[9458,7819,8436,7758,7568,7643,8591,8349,7515,7876,8126,9074]},{name:"4",data:[17668,14918,15667,14730,14316,14569,16027,15796,14338,14902,14670,16745]},{name:"5",data:[67782,58541,62850,56459,54246,54163,60865,60382,55466,56339,55773,63557]}],reviewCountPlot:{dataLabels:{enabled:!0},chart:{type:"bar",height:350,stacked:!0,stackType:"100%",toolbar:{show:!1}},plotOptions:{bar:{horizontal:!0}},stroke:{width:1,colors:["#fff"]},xaxis:{categories:["Verified","Unverified"]},fill:{opacity:1},legend:{position:"top",horizontalAlign:"left",offsetX:40}},chartOptions:{dataLabels:{enabled:!1},yaxis:{labels:{style:{fontSize:"18px"}},decimalsInFloat:1},chart:{toolbar:{show:!1}},xaxis:{labels:{style:{fontSize:"18px"}},decimalsInFloat:1,categories:["B005NF5NTK","B0092KJ9BU","B00AANQLRI","B00BT8L2MW","B00D856NOG","B00G7UY3EG","B00IGISUTG","B00M51DDT2","B00M6QODH2","B00MQSMDYU","B00MXWFUQC","B00P7N0320","B00QN1T6NM","B00UCZGS6S","B00UH3L82Y","B00VH88CJ0","B00X5RV14Y","B014EB532U","B018JW3EOY","B019PV2I3G"]}},count:[[{data:[47,5],name:1},{data:[32,2],name:2},{data:[79,5],name:3},{data:[231,9],name:4},{data:[1189,35],name:5}],[{data:[64,8],name:1},{data:[72,9],name:2},{data:[90,17],name:3},{data:[155,28],name:4},{data:[580,41],name:5}],[{data:[22,3],name:1},{data:[14,1],name:2},{data:[37,4],name:3},{data:[141,2],name:4},{data:[774,25],name:5}],[{data:[35,3],name:1},{data:[31,3],name:2},{data:[35,1],name:3},{data:[93,3],name:4},{data:[800,24],name:5}],[{data:[11,2],name:1},{data:[14,1],name:2},{data:[38,1],name:3},{data:[139,12],name:4},{data:[773,50],name:5}],[{data:[132,11],name:1},{data:[41,0],name:2},{data:[48,1],name:3},{data:[120,11],name:4},{data:[942,71],name:5}],[{data:[40,2],name:1},{data:[26,0],name:2},{data:[36,0],name:3},{data:[103,2],name:4},{data:[892,18],name:5}],[{data:[40,5],name:1},{data:[21,6],name:2},{data:[41,3],name:3},{data:[120,8],name:4},{data:[954,69],name:5}],[{data:[33,7],name:1},{data:[24,7],name:2},{data:[38,5],name:3},{data:[89,26],name:4},{data:[592,156],name:5}],[{data:[39,2],name:1},{data:[19,3],name:2},{data:[41,2],name:3},{data:[134,10],name:4},{data:[1006,84],name:5}],[{data:[97,7],name:1},{data:[70,3],name:2},{data:[83,2],name:3},{data:[148,14],name:4},{data:[484,70],name:5}],[{data:[24,3],name:1},{data:[19,0],name:2},{data:[41,1],name:3},{data:[154,13],name:4},{data:[1109,149],name:5}],[{data:[62,4],name:1},{data:[46,2],name:2},{data:[67,6],name:3},{data:[162,22],name:4},{data:[754,102],name:5}],[{data:[75,8],name:1},{data:[46,5],name:2},{data:[70,3],name:3},{data:[161,8],name:4},{data:[609,23],name:5}],[{data:[152,6],name:1},{data:[81,2],name:2},{data:[121,4],name:3},{data:[137,10],name:4},{data:[467,26],name:5}],[{data:[25,2],name:1},{data:[20,1],name:2},{data:[29,4],name:3},{data:[169,8],name:4},{data:[1355,69],name:5}],[{data:[14,5],name:1},{data:[11,1],name:2},{data:[31,3],name:3},{data:[120,18],name:4},{data:[1133,177],name:5}],[{data:[80,4],name:1},{data:[45,1],name:2},{data:[60,0],name:3},{data:[121,5],name:4},{data:[668,17],name:5}],[{data:[23,3],name:1},{data:[18,0],name:2},{data:[41,1],name:3},{data:[156,12],name:4},{data:[1111,141],name:5}],[{data:[37,5],name:1},{data:[39,3],name:2},{data:[36,0],name:3},{data:[102,4],name:4},{data:[1216,66],name:5}]],series:[{name:"verified",data:[4.573510773130545,4.160249739854319,4.65080971659919,4.601609657947686,4.691282051282052,4.324240062353858,4.62351868732908,4.63860544217687,4.524484536082475,4.653753026634383,3.9659863945578233,4.711210096510765,4.374885426214482,4.231009365244537,3.7160751565762005,4.757822277847309,4.792971734148205,4.285420944558521,4.715344699777613,4.693006993006993]},{name:"unverified",data:[4.196428571428571,3.825242718446602,4.285714285714286,4.235294117647059,4.621212121212121,4.3936170212765955,4.545454545454546,4.428571428571429,4.577114427860696,4.693069306930693,4.427083333333333,4.837349397590361,4.588235294117647,3.702127659574468,4,4.678571428571429,4.769607843137255,4.111111111111111,4.834394904458598,4.576923076923077]},{name:"all",data:[4.560587515299877,4.12781954887218,4.638318670576735,4.589494163424124,4.686839577329491,4.328976034858388,4.621983914209116,4.623520126282557,4.535312180143296,4.656716417910448,4.011247443762781,4.725049570389953,4.398533007334963,4.2063492063492065,3.7296222664015906,4.7538644470868014,4.789821546596166,4.280719280719281,4.727755644090306,4.687002652519894]}],lda:[{name:"TaoTronics Car Phone Mount Holder, Windshield /",description:"Dashboard Universal Car Mobile Phone cradle for iOS / Android Smartphone and More",code:"B00MXWFUQC"},{name:"ArmorSuit MilitaryShield Max Coverage Screen Protector for Apple Watch",description:"42mm (Series 3 / 2 / 1 Compatible) [2 Pack] - Anti-Bubble HD Clear Film",code:"B00UH3L82Y"},{name:"Portable Charger Anker PowerCore 20100mAh",description:"Ultra High Capacity Power Bank with 4.8A Output and PowerIQ Technology.",code:"B00X5RV14Y"},{name:"Anker 24W Dual USB Car Charger",description:"PowerDrive 2 for iPhone X / 8/7 / 6s / Plus, iPad Pro/Air 2 / Mini, Note 5/4, LG, Nexus, HTC, and More",code:"B00VH88CJ0"},{name:"Anker Astro E1 5200mAh Candy bar-Sized Ultra Compact Portable Charger",description:"(External Battery Power Bank) with High-Speed Charging PowerIQ Technology",code:"B018JW3EOY"},{name:"Plantronics Voyager Legend Wireless Bluetooth Headset",description:"Compatible with iPhone, Android, and Other Leading Smartphones - Black",code:"B0092KJ9BU"},{name:"Portable chargers",description:"Aggregated LDA",code:"PORTABLECHARGERS"}],ecosystem:[]})},l=s,c=(t("dd78"),t("2877")),m=t("6544"),x=t.n(m),y=t("7496"),p=t("8336"),u=t("b0af"),h=t("99d9"),f=t("62ad"),g=t("a523"),v=t("a75b"),b=t("0e8f"),w=t("132d"),P=t("adda"),_=t("a722"),S=t("0fd9"),B=t("b974"),C=t("71a3"),k=t("fe57"),A=t("8654"),T=Object(c["a"])(l,i,d,!1,null,null,null),O=T.exports;x()(T,{VApp:y["a"],VBtn:p["a"],VCard:u["a"],VCardActions:h["a"],VCardSubtitle:h["b"],VCardTitle:h["c"],VCol:f["a"],VContainer:g["a"],VContent:v["a"],VFlex:b["a"],VIcon:w["a"],VImg:P["a"],VLayout:_["a"],VRow:S["a"],VSelect:B["a"],VTab:C["a"],VTabs:k["a"],VTextField:A["a"]});var V={name:"App",components:{Demo:O},data:()=>({})},M=V,D=(t("034f"),Object(c["a"])(M,o,r,!1,null,null,null)),U=D.exports,L=t("f309");n["a"].use(L["a"]);var j=new L["a"]({}),E=t("bc3a"),I=t.n(E),z=t("1321"),N=t.n(z);t("5363"),t("b56c"),t("d244"),t("d5e8");n["a"].config.productionTip=!1,n["a"].prototype.$axios=I.a,n["a"].use(N.a),n["a"].component("apexchart",N.a),new n["a"]({vuetify:j,render:e=>e(U)}).$mount("#app")},"8a23":function(e,a,t){},b56c:function(e,a,t){},dd78:function(e,a,t){"use strict";var n=t("e91c"),o=t.n(n);o.a},e91c:function(e,a,t){}});
//# sourceMappingURL=app.52c1be34.js.map