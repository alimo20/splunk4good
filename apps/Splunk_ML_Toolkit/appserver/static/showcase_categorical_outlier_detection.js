webpackJsonp([13],[function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__webpack_require__.p=function(){function make_url(){for(var seg,len,output="",i=0,l=arguments.length;i<l;i++)seg=arguments[i].toString(),len=seg.length,len>1&&"/"==seg.charAt(len-1)&&(seg=seg.substring(0,len-1)),output+="/"!=seg.charAt(0)?"/"+seg:seg;if("/"!=output){var segments=output.split("/"),firstseg=segments[1];if("static"==firstseg||"modules"==firstseg){var postfix=output.substring(firstseg.length+2,output.length);output="/"+firstseg,window.$C.BUILD_NUMBER&&(output+="/@"+window.$C.BUILD_NUMBER),window.$C.BUILD_PUSH_NUMBER&&(output+="."+window.$C.BUILD_PUSH_NUMBER),"app"==segments[2]&&(output+=":"+getConfigValue("APP_BUILD",0)),output+="/"+postfix}}var root=getConfigValue("MRSPARKLE_ROOT_PATH","/"),djangoRoot=getConfigValue("DJANGO_ROOT_PATH",""),locale=getConfigValue("LOCALE","en-US"),combinedPath="";return combinedPath=djangoRoot&&output.substring(0,djangoRoot.length)===djangoRoot?output.replace(djangoRoot,djangoRoot+"/"+locale.toLowerCase()):"/"+locale+output,""==root||"/"==root?combinedPath:root+combinedPath}function getConfigValue(key,defaultValue){if(window.$C&&window.$C.hasOwnProperty(key))return window.$C[key];if(void 0!==defaultValue)return defaultValue;throw new Error("getConfigValue - "+key+" not set, no default provided")}return make_url("/static/app/Splunk_ML_Toolkit/")+"/"}(),__WEBPACK_AMD_DEFINE_ARRAY__=[__webpack_require__("shim/jquery"),__webpack_require__(513),__webpack_require__("splunkjs/mvc/singleview"),__webpack_require__("splunkjs/mvc/tableview"),__webpack_require__(620),__webpack_require__(624),__webpack_require__(625),__webpack_require__(621),__webpack_require__(547),__webpack_require__(626),__webpack_require__(538),__webpack_require__(628),__webpack_require__(541),__webpack_require__(629),__webpack_require__(630),__webpack_require__(631)],__WEBPACK_AMD_DEFINE_RESULT__=function(_jquery,_underscoreMltk,_singleview,_tableview,_Master,_DrilldownLinker,_SearchStringDisplay,_Spinners,_compactTemplateString,_ColorPalette,_AnomalyDetection,_EnhancedMultiDropdownView,_Forms,_Searches,_TableUtils,_BaseAssistantView){"use strict";function _interopRequireWildcard(obj){if(obj&&obj.__esModule)return obj;var newObj={};if(null!=obj)for(var key in obj)Object.prototype.hasOwnProperty.call(obj,key)&&(newObj[key]=obj[key]);return newObj.default=obj,newObj}function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function _taggedTemplateLiteral(strings,raw){return Object.freeze(Object.defineProperties(strings,{raw:{value:Object.freeze(raw)}}))}var _jquery2=_interopRequireDefault(_jquery),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_singleview2=_interopRequireDefault(_singleview),_tableview2=_interopRequireDefault(_tableview),_Master2=_interopRequireDefault(_Master),DrilldownLinker=_interopRequireWildcard(_DrilldownLinker),Spinners=_interopRequireWildcard(_Spinners),_compactTemplateString2=_interopRequireDefault(_compactTemplateString),_AnomalyDetection2=_interopRequireDefault(_AnomalyDetection),_EnhancedMultiDropdownView2=_interopRequireDefault(_EnhancedMultiDropdownView),Forms=_interopRequireWildcard(_Forms),Searches=_interopRequireWildcard(_Searches),TableUtils=_interopRequireWildcard(_TableUtils),_BaseAssistantView2=_interopRequireDefault(_BaseAssistantView),_slicedToArray=function(){function sliceIterator(arr,i){var _arr=[],_n=!0,_d=!1,_e=void 0;try{for(var _s,_i=arr[Symbol.iterator]();!(_n=(_s=_i.next()).done)&&(_arr.push(_s.value),!i||_arr.length!==i);_n=!0);}catch(err){_d=!0,_e=err}finally{try{!_n&&_i.return&&_i.return()}finally{if(_d)throw _e}}return _arr}return function(arr,i){if(Array.isArray(arr))return arr;if(Symbol.iterator in Object(arr))return sliceIterator(arr,i);throw new TypeError("Invalid attempt to destructure non-iterable instance")}}(),_templateObject=_taggedTemplateLiteral(['| loadjob $searchBarSearchJobIdToken$\n                               | head 1\n                               | transpose\n                               | table column\n                               | search column != "column" AND column != "_*"'],['| loadjob $searchBarSearchJobIdToken$\n                               | head 1\n                               | transpose\n                               | table column\n                               | search column != "column" AND column != "_*"']),_templateObject2=_taggedTemplateLiteral(["| loadjob $searchBarSearchJobIdToken$ | stats count"],["| loadjob $searchBarSearchJobIdToken$ | stats count"]),CategoricalOutlierDetectionView=_BaseAssistantView2.default.extend({headerOptions:{title:"Detect Categorical Outliers",description:"Find events that contain unusual combinations of values."},submitButtonText:"Detect Outliers",historyStatisticsFields:["outlierCount"],activateSaveTooltip:"To activate Save, successfully detect outliers.",renderPanels:function(){function getVizQuery(sharedSearchArray){var searchInfo=self.getSearchInfo(),vizQueryArray=[searchInfo.searchString].concat(sharedSearchArray),vizQuerySearch=DrilldownLinker.createSearch(vizQueryArray,searchInfo.timeRange);return[vizQueryArray,vizQuerySearch]}function updateForm(newIsRunningValue){if(null!=newIsRunningValue&&(isRunning=newIsRunningValue),self.controls.anomalyFieldsControl.settings.set("disabled",isRunning),isRunning)self.model.state.set({footerDisabled:isRunning,submitButtonText:"Detecting Outliers..."}),submitted&&self.model.state.trigger("submitStarted");else{var anomalyFieldsToken=Forms.getToken("anomalyFieldsToken"),fieldsValid=null!=anomalyFieldsToken&&anomalyFieldsToken.length>0;self.model.state.set({footerDisabled:!fieldsValid,submitButtonText:self.submitButtonText}),fieldsValid&&submitted&&(submitSuccess&&(self.model.experimentSubmitPristine.setFromSplunkD(self.model.experiment.toSplunkD()),self.model.state.trigger("submitSuccess"),submitSuccess=!1),submitted=!1)}}var _this=this,self=this;_BaseAssistantView2.default.prototype.renderPanels.call(this);var isRunning=!1,submitted=!1,submitSuccess=!1;this.controls.anomalyFieldsControl=function(){var control=new _EnhancedMultiDropdownView2.default({id:"anomalyFieldsControl",managerid:"anomalyFieldsSearch",el:_this.$el.find("#anomalyFieldsControl"),labelField:"column",valueField:"column",width:400});return control.$el.prev("label").tooltip({title:"Select the fields to consider. Events with fields taking on rare values, especially events with multiple such fields, may be considered outliers."}),control.on("change",function(){var fields=control.val();self.model.experiment.getMainSearchStageModel().fields.resetFields(fields),null!=fields&&fields.length>0?(Forms.setToken("rawAnomalyFieldsToken",fields),Forms.setToken("anomalyFieldsToken",fields.map(Forms.escape).join(" "))):Forms.unsetToken("anomalyFieldsToken"),updateForm()}),control.render(),control}();var singleOutliersPanel=function(){return new _Master2.default({el:_this.$el.find("#singleOutliersPanel"),title:"Outlier(s)",tooltip:"Number of events that are outliers.",viz:_singleview2.default,vizOptions:{id:"singleOutliersViz",managerid:"anomalousEventsCountSearch",underLabel:"Outlier(s)"}})}(),singleResultsPanel=function(){return new _Master2.default({el:_this.$el.find("#singleResultsPanel"),title:"Total Event(s)",viz:_singleview2.default,vizOptions:{id:"singleResultsViz",managerid:"anomalyDetectionResultsCountSearch",underLabel:"Total Event(s)"}})}(),outliersTablePanel=function(){var assistantPanel=new _Master2.default({el:_this.$el.find("#outliersTablePanel"),title:"Data and Outliers",tooltip:"The input events with added fields to indicate whether each event is an outlier (isOutlier) and the field that most strongly contributed to this classification (probable_cause).",viz:_tableview2.default,vizOptions:{id:"outliersTable",managerid:"anomalyDetectionResultsSearch",drilldown:"none"}}),outlierFieldIndexArray=[],fieldsCache=[],HighlightedTableRender=_tableview2.default.BaseCellRenderer.extend({canRender:function(){return!0},teardown:function(){this.outlierEventTriggered=!1},render:function($td,cell){if(fieldsCache.push(cell.field),$td.text(cell.value),"probable_cause"===cell.field&&null!=cell.value)outlierFieldIndexArray.push(fieldsCache.indexOf(cell.value)),$td.addClass("outlier-event");else if("isOutlier"===cell.field){fieldsCache=[];var icon="check",colorIndex=7;"1"===cell.value&&(icon="alert",colorIndex=1,this.outlierEventTriggered||(assistantPanel.viz.trigger("outlierFound"),this.outlierEventTriggered=!0)),$td.addClass("icon-inline").html(_underscoreMltk2.default.template('<i class="icon-<%-icon%>" style="color: <%-color%>"></i> &#160 <%- text %> ',{icon:icon,text:cell.value,color:(0,_ColorPalette.getColorByIndex)(colorIndex)}))}$td.addClass(TableUtils.columnTypeToClassName(cell.columnType))}});return assistantPanel.viz.addCellRenderer(new HighlightedTableRender),assistantPanel.viz.on("outlierFound",function(){_underscoreMltk2.default.defer(function(){assistantPanel.viz.$el.find("td.outlier-event.string").each(function(index){if(null!=outlierFieldIndexArray[index]){var fieldIndex=outlierFieldIndexArray[index];(0,_jquery2.default)(this).parents("tr").find("td:eq("+fieldIndex+")").css("background-color",(0,_ColorPalette.getColorByIndex)(1))}}),outlierFieldIndexArray=[]})}),assistantPanel}();return this.assistantFormView.searchBarControl.events.on("change",function(){Forms.clearChoiceView(self.controls.anomalyFieldsControl,!0),Forms.unsetToken("anomalyFieldsToken","anomalyDetectionResultsToken"),updateForm()}),this.listenTo(self.model.state,"submit",function(){Searches.startSearch("anomalyDetectionResultsSearch")}),function(){var searchBarSearch=Searches.getSearchManager("searchBarSearch");searchBarSearch.on("onStartCallback",function(){self.hideErrorMessage(),self.hideResults()}),searchBarSearch.on("onErrorCallback",function(errorMessage){self.showErrorMessage(errorMessage),self.hideResults()})}(),function(){Searches.setSearch("anomalyFieldsSearch",{searchString:(0,_compactTemplateString2.default)(_templateObject),onStartCallback:function(){self.hideErrorMessage()},onErrorCallback:function(errorMessage){self.showErrorMessage(errorMessage),self.hideResults()}})}(),function(){function openInSearch(){var _getVizQuery=getVizQuery(sharedSearchArray),_getVizQuery2=_slicedToArray(_getVizQuery,2);vizQueryArray=_getVizQuery2[0],vizQuerySearch=_getVizQuery2[1],window.open(DrilldownLinker.getUrl("search",vizQuerySearch),"_blank")}function showSPL(){var searchInfo=self.getSearchInfo(),_getVizQuery3=getVizQuery(sharedSearchArray),_getVizQuery4=_slicedToArray(_getVizQuery3,2);vizQueryArray=_getVizQuery4[0],vizQuerySearch=_getVizQuery4[1],(0,_SearchStringDisplay.showSearchStringModal)("anomalyDetectionResultsSearchModal","Display the outliers in search",vizQueryArray,[null,"compute the categorical outliers","add a field to identify the outliers","reorder the fields","sort the results to make outliers appear at the top"],searchInfo.timeRange)}var sharedSearchArray=["| anomalydetection $anomalyFieldsToken$ $anomalyDetectionParamsToken$",'| eval isOutlier = if(probable_cause != "", "1", "0")',"| table $anomalyFieldsToken$, probable_cause, isOutlier","| sort 100000 probable_cause"],vizQueryArray=[],vizQuerySearch=null;self.listenTo(self.model.state,"openInSearch",openInSearch),self.listenTo(self.model.state,"showSPL",showSPL),outliersTablePanel.openInSearchButton.on("click",openInSearch),outliersTablePanel.showSPLButton.on("click",showSPL),Searches.setSearch("anomalyDetectionResultsSearch",{targetJobIdTokenName:"anomalyDetectionResultsToken",autostart:!1,searchString:["| loadjob $searchBarSearchJobIdToken$"].concat(sharedSearchArray),onStartCallback:function(){self.hideResults(),submitted=!0,updateForm(!0);var _getVizQuery5=getVizQuery(sharedSearchArray),_getVizQuery6=_slicedToArray(_getVizQuery5,2);vizQueryArray=_getVizQuery6[0],vizQuerySearch=_getVizQuery6[1],DrilldownLinker.setSearchDrilldown(outliersTablePanel.title,vizQuerySearch)},onDoneCallback:function(){self.accumulateExperimentHistory(Searches.getSid(this)),self.showResults(),submitSuccess=!0},onFinallyCallback:function(){updateForm(!1)}})}(),function(){var vizQueryArray=[],vizQuerySearch=null,sharedStatsString="| stats count as outlierCount",vizOptions=DrilldownLinker.parseVizOptions({category:"singlevalue"});singleOutliersPanel.openInSearchButton.on("click",function(){window.open(DrilldownLinker.getUrl("search",vizQuerySearch,vizOptions),"_blank")}),singleOutliersPanel.showSPLButton.on("click",function(){var searchInfo=self.getSearchInfo();(0,_SearchStringDisplay.showSearchStringModal)("anomalousEventsCountSearchModal","Display the number of outliers",vizQueryArray,[null,"compute the categorical outliers","count the outliers"],searchInfo.timeRange,vizOptions)}),Searches.setSearch("anomalousEventsCountSearch",{targetJobIdTokenName:"anomalousEventsCountToken",searchString:"| loadjob $anomalyDetectionResultsToken$ | where isOutlier=1 "+sharedStatsString,onStartCallback:function(){Spinners.showLoadingOverlay(singleOutliersPanel.viz.$el);var _getVizQuery7=getVizQuery(["| anomalydetection $anomalyFieldsToken$",sharedStatsString]),_getVizQuery8=_slicedToArray(_getVizQuery7,2);vizQueryArray=_getVizQuery8[0],vizQuerySearch=_getVizQuery8[1],DrilldownLinker.setSearchDrilldown(singleOutliersPanel.title,vizQuerySearch,vizOptions)},onDataCallback:function(data){self.accumulateExperimentStatistics(Searches.getSid("anomalyDetectionResultsSearch"),data)},onFinallyCallback:function(){Spinners.hideLoadingOverlay(singleOutliersPanel.viz.$el)}})}(),function(){var vizQueryArray=[],vizQuerySearch=null,searchInfo=self.getSearchInfo(),vizOptions=DrilldownLinker.parseVizOptions({category:"singlevalue"});singleResultsPanel.openInSearchButton.on("click",function(){window.open(DrilldownLinker.getUrl("search",vizQuerySearch,vizOptions),"_blank")}),singleResultsPanel.showSPLButton.on("click",function(){(0,_SearchStringDisplay.showSearchStringModal)("anomalyDetectionResultsCountSearchModal","Display the number of results",vizQueryArray,[null,"annotate the results with categorical outliers","count the results"],searchInfo.timeRange,vizOptions)}),Searches.setSearch("anomalyDetectionResultsCountSearch",{targetJobIdTokenName:"anomalyDetectionResultsCountToken",searchString:(0,_compactTemplateString2.default)(_templateObject2),onStartCallback:function(){Spinners.showLoadingOverlay(singleResultsPanel.viz.$el);var _getVizQuery9=getVizQuery(["| stats count"]),_getVizQuery10=_slicedToArray(_getVizQuery9,2);vizQueryArray=_getVizQuery10[0],vizQuerySearch=_getVizQuery10[1],DrilldownLinker.setSearchDrilldown(singleResultsPanel.title,vizQuerySearch,vizOptions)},onFinallyCallback:function(data){Spinners.hideLoadingOverlay(singleResultsPanel.viz.$el)}})}(),setTimeout(updateForm,0),this},loadExperiment:function(runExperiment){var _this2=this;this.stopListening(this.controls.anomalyFieldsControl,"datachange");var searchInfo=this.getSearchInfo();this.assistantFormView.searchBarControl.setSearchInfo(searchInfo);var mainSearchStage=this.model.experiment.getMainSearchStageModel();if(null!=mainSearchStage){var paramsArray=mainSearchStage.params.getParamsArray(),anomalyFields=mainSearchStage.fields.getFieldsArray();Forms.setToken("anomalyDetectionParamsToken",paramsArray.join(" ")),this.listenToOnce(this.controls.anomalyFieldsControl,"datachange",_underscoreMltk2.default.partial(function(fields,runExp){Forms.setChoiceViewValueIfValid(_this2.controls.anomalyFieldsControl,fields),runExp&&_this2.model.state.trigger("submit")},anomalyFields,runExperiment))}},setDefaultMainStage:function(){var mainSearchStage=this.model.experiment.getMainSearchStageModel();null==mainSearchStage&&(mainSearchStage=new _AnomalyDetection2.default,this.model.experiment.addSearchStages(mainSearchStage)),mainSearchStage.params.resetParams({action:"annotate"})},controlsTemplate:'\n        <div class="mlts-input">\n            <label>Field(s) to analyze</label>\n            <div id="anomalyFieldsControl"></div>\n        </div>\n    ',template:'\n        <div class="mlts-row mlts-results-row">\n            <div class="mlts-cell">\n                <div class="mlts-panel" id="singleOutliersPanel"></div>\n            </div>\n            <div class="mlts-cell">\n                <div class="mlts-panel"  id="singleResultsPanel"></div>\n            </div>\n        </div>\n        <div class="mlts-row mlts-results-row">\n            <div class="mlts-cell">\n                <div class="mlts-panel" id="outliersTablePanel"></div>\n            </div>\n        </div>\n    '});new CategoricalOutlierDetectionView}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}]);