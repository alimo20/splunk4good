webpackJsonp([10],{0:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__webpack_require__.p=function(){function make_url(){for(var seg,len,output="",i=0,l=arguments.length;i<l;i++)seg=arguments[i].toString(),len=seg.length,len>1&&"/"==seg.charAt(len-1)&&(seg=seg.substring(0,len-1)),output+="/"!=seg.charAt(0)?"/"+seg:seg;if("/"!=output){var segments=output.split("/"),firstseg=segments[1];if("static"==firstseg||"modules"==firstseg){var postfix=output.substring(firstseg.length+2,output.length);output="/"+firstseg,window.$C.BUILD_NUMBER&&(output+="/@"+window.$C.BUILD_NUMBER),window.$C.BUILD_PUSH_NUMBER&&(output+="."+window.$C.BUILD_PUSH_NUMBER),"app"==segments[2]&&(output+=":"+getConfigValue("APP_BUILD",0)),output+="/"+postfix}}var root=getConfigValue("MRSPARKLE_ROOT_PATH","/"),djangoRoot=getConfigValue("DJANGO_ROOT_PATH",""),locale=getConfigValue("LOCALE","en-US"),combinedPath="";return combinedPath=djangoRoot&&output.substring(0,djangoRoot.length)===djangoRoot?output.replace(djangoRoot,djangoRoot+"/"+locale.toLowerCase()):"/"+locale+output,""==root||"/"==root?combinedPath:root+combinedPath}function getConfigValue(key,defaultValue){if(window.$C&&window.$C.hasOwnProperty(key))return window.$C[key];if(void 0!==defaultValue)return defaultValue;throw new Error("getConfigValue - "+key+" not set, no default provided")}return make_url("/static/app/Splunk_ML_Toolkit/")+"/"}(),__WEBPACK_AMD_DEFINE_ARRAY__=[__webpack_require__("shim/jquery"),__webpack_require__(513),__webpack_require__("models/Base"),__webpack_require__("models/services/data/ui/Pref"),__webpack_require__("models/shared/ClassicURL"),__webpack_require__("splunkjs/mvc/sharedmodels"),__webpack_require__("util/splunkd_utils"),__webpack_require__(604),__webpack_require__(535),__webpack_require__(511),__webpack_require__(567),__webpack_require__(568),__webpack_require__(569),__webpack_require__(570),__webpack_require__(603)],__WEBPACK_AMD_DEFINE_RESULT__=function(_jquery,_underscoreMltk,_Base,_Pref,_ClassicURL,_sharedmodels,_splunkd_utils,_Experiments,_ExperimentModel,_BaseDashboardView,_Create,_TypeFilter,_TableCaption,_Master,_TypeSelector){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function _toConsumableArray(arr){if(Array.isArray(arr)){for(var i=0,arr2=Array(arr.length);i<arr.length;i++)arr2[i]=arr[i];return arr2}return Array.from(arr)}function _defineProperty(obj,key,value){return key in obj?Object.defineProperty(obj,key,{value:value,enumerable:!0,configurable:!0,writable:!0}):obj[key]=value,obj}var _jquery2=_interopRequireDefault(_jquery),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_Base2=_interopRequireDefault(_Base),_Pref2=_interopRequireDefault(_Pref),_ClassicURL2=_interopRequireDefault(_ClassicURL),_sharedmodels2=_interopRequireDefault(_sharedmodels),_splunkd_utils2=_interopRequireDefault(_splunkd_utils),_Experiments2=_interopRequireDefault(_Experiments),_ExperimentModel2=_interopRequireDefault(_ExperimentModel),_BaseDashboardView2=_interopRequireDefault(_BaseDashboardView),_Create2=_interopRequireDefault(_Create),_TypeFilter2=_interopRequireDefault(_TypeFilter),_TableCaption2=_interopRequireDefault(_TableCaption),_Master2=_interopRequireDefault(_Master),_TypeSelector2=_interopRequireDefault(_TypeSelector),experimentsListContainerId="experimentsListContainer",experimentButtonClass="mltk-add-experiment",captionFilterKey=["title"],ExperimentsView=_BaseDashboardView2.default.extend({headerOptions:{title:"Experiments",hidePrintButton:!0,extraButtons:[(0,_jquery2.default)('<button class="btn btn-primary '+experimentButtonClass+'">Create New Experiment</button>')]},events:_defineProperty({},"click ."+experimentButtonClass,function(e){var _this=this,experiment=new _ExperimentModel2.default;if(experiment.app=this.model.application.get("app"),experiment.owner=this.model.application.get("owner"),!this.fetchingExperiment){this.fetchingExperiment=!0,this.$("."+experimentButtonClass).addClass("disabled");var experimentDeferred=experiment.fetch();_jquery2.default.when(experimentDeferred).always(function(){_this.fetchingExperiment=!1,_this.children.createExperimentModal=new _Create2.default({experimentType:_this.model.state.get("experimentType"),experimentTypes:ExperimentsView.SORTED_EXPERIMENT_TYPES,showExperimentTypePicker:!0,model:{experiment:experiment,application:_this.model.application},onHiddenRemove:!0}),_this.children.createExperimentModal.render().appendTo((0,_jquery2.default)("body")).show(),_this.listenToOnce(_this.children.createExperimentModal,"hidden",function(){_this.$("."+experimentButtonClass).removeClass("disabled")})})}e.preventDefault()}),initialize:function(options){var _this2=this;_BaseDashboardView2.default.prototype.initialize.call(this,options),this.collection=this.collection||{},this.model=this.model||{},this.collection.activeExperiments=new _Experiments2.default,this.experimentsCollections={},this.experimentsCountModel=new _Base2.default,this.fetchingExperiment=!1,ExperimentsView.SORTED_EXPERIMENT_TYPES.forEach(function(type){_this2.experimentsCollections[type]=new _Experiments2.default({experimentType:type}),_this2.experimentsCountModel.set(type,"")}),this.model.uiPrefs=new _Pref2.default,this.model.classicURL=new _ClassicURL2.default,this.model.application=_sharedmodels2.default.get("app"),this.model.stateDefaults={sortKey:"name",sortDirection:"asc",offset:0,search:_splunkd_utils2.default.createSearchFilterString("",captionFilterKey)},this.model.state=new _Base2.default(_underscoreMltk2.default.extend({count:100,fetching:!0,experimentPageRenderred:!1},this.model.stateDefaults)),this.syncFromClassicUrl(),this.rawSearch=new _Base2.default,this.deferreds.uiPrefs=_jquery2.default.Deferred(),this.model.uiPrefs.bootstrap(this.deferreds.uiPrefs,this.model.application.get("page"),this.model.application.get("app"),this.model.application.get("owner")),this.model.uiPrefs.entry.content.on("change",function(){_this2.populateUIPrefs()}),this.model.uiPrefs.entry.content.on("change:display.prefs.aclFilter",function(){_this2.fetchListCollection()}),this.children.experimentTypeFilterView=new _TypeFilter2.default({experimentTypes:ExperimentsView.SORTED_EXPERIMENT_TYPES,model:{stateModel:this.model.state,countModel:this.experimentsCountModel}}),this.experimentsTypeSelectorView=new _TypeSelector2.default({experimentTypes:ExperimentsView.SORTED_EXPERIMENT_TYPES,model:{experiment:this.model.experiment,application:this.model.application}}),this.initializeListCollections()},populateUIPrefs:function(){var data={};this.model.uiPrefs.isNew()&&(data.app=this.model.application.get("app"),data.owner=this.model.application.get("owner")),this.model.uiPrefs.save({},{data:data})},syncFromClassicUrl:function(){this.model.classicURL.fetch();var experimentTypeFromURL=this.model.classicURL.get("experimentType");experimentTypeFromURL&&ExperimentsView.SORTED_EXPERIMENT_TYPES.indexOf(experimentTypeFromURL)!==-1?this.model.state.set("experimentType",experimentTypeFromURL):this.model.state.set("experimentType",ExperimentsView.SORTED_EXPERIMENT_TYPES[0])},getButtonFilterSearch:function(){var buttonFilter=this.model.uiPrefs.entry.content.get("display.prefs.aclFilter");if(_underscoreMltk2.default.isUndefined(buttonFilter)||"none"===buttonFilter)return"";switch(buttonFilter){case"owner":return"(eai:acl.owner="+_splunkd_utils2.default.quoteSearchFilterValue(this.model.application.get("owner"))+")";case"app":return"(eai:acl.app="+_splunkd_utils2.default.quoteSearchFilterValue(this.model.application.get("app"))+")";default:return""}},getCollectionFilters:function(){var app="system"===this.model.application.get("app")?"-":this.model.application.get("app"),search=this.model.state.get("search")||"",buttonFilterSearch=this.getButtonFilterSearch(),sortDir=this.model.state.get("sortDirection"),sortKey=this.model.state.get("sortKey").split(","),sortMode="natural";return buttonFilterSearch&&(search+=buttonFilterSearch),"name"!==sortKey[0]&&"eai:acl.owner"!==sortKey[0]&&"eai:acl.app"!==sortKey[0]&&"eai:acl.sharing"!==sortKey[0]||(sortDir=[sortDir,sortDir],sortMode=[sortMode,sortMode]),{app:app,owner:_sharedmodels2.default.get("app").get("owner"),sort_dir:sortDir,sort_key:sortKey,sort_mode:sortMode,search:search,count:this.model.state.get("count"),offset:this.model.state.get("offset")}},getTotalCount:function(){var _this3=this,reducer=function(accumulator,currentValue){return accumulator+currentValue};return ExperimentsView.SORTED_EXPERIMENT_TYPES.map(function(type){return _this3.experimentsCountModel.get(type)||0}).reduce(reducer)},initializeListCollections:function(){var _this4=this,filters=_underscoreMltk2.default.extend({count:1},this.getCollectionFilters());this.model.state.set("fetching",!0);var countFetchDeferreds=ExperimentsView.SORTED_EXPERIMENT_TYPES.map(function(key){var countDeferred=_jquery2.default.Deferred();return _this4.experimentsCollections[key].fetch({data:filters,success:function(fetchedCollection){_this4.experimentsCountModel.set(fetchedCollection.experimentType,fetchedCollection.paging.get("total")),countDeferred.resolve(),fetchedCollection===_this4.experimentsCollections[_this4.model.state.get("experimentType")]&&_this4.updateListCollection()}}),countDeferred});_jquery2.default.when.apply(_jquery2.default,_toConsumableArray(countFetchDeferreds)).then(function(){_this4.manageStateOfChildren(),_this4.experimentsCountModel.on("change",function(){_this4.manageStateOfChildren()})})},fetchListCollection:function(){var _this5=this,filters=this.getCollectionFilters(),type=this.model.state.get("experimentType");return this.model.state.set("fetching",!0),this.experimentsCollections[type].fetch({data:filters,success:function(fetchedCollection){filters.search===_splunkd_utils2.default.createSearchFilterString("",captionFilterKey)&&_this5.experimentsCountModel.set(fetchedCollection.experimentType,fetchedCollection.paging.get("total")),_this5.updateListCollection()}})},updateListCollection:function(){this.model.state.set("fetching",!1);var type=this.model.state.get("experimentType");this.collection.activeExperiments.links=this.experimentsCollections[type].links,this.collection.activeExperiments.paging=this.experimentsCollections[type].paging,this.collection.activeExperiments.reset(this.experimentsCollections[type].models)},manageStateOfChildren:function(){this.model.state.get("experimentPageRenderred")&&(0===this.getTotalCount()?(this.headerView.$el.hide(),this.children.experimentTypeFilterView.$el.hide(),this.experimentsList$El.hide(),this.experimentsTypeSelectorView.$el.show()):(this.experimentsTypeSelectorView.$el.hide(),this.headerView.$el.css("display",""),this.children.experimentTypeFilterView.$el.css("display",""),this.experimentsList$El.css("display","")))},render:function(){var _this6=this;return _BaseDashboardView2.default.prototype.render.call(this),this.experimentsTypeSelectorView.render().appendTo(this.$el),this.experimentsList$El=this.$el.find("#"+experimentsListContainerId),this.children.experimentTypeFilterView.render().$el.insertBefore(this.experimentsList$El),this.model.state.on("change:sortDirection change:sortKey change:search change:offset change:experimentType",_underscoreMltk2.default.debounce(function(model){_this6.model.state.changed.experimentType&&(_this6.model.state.set(_this6.model.stateDefaults,{silent:!0}),_this6.children.caption.children.input.clear()),_this6.fetchListCollection()},0),this),this.model.state.on("change:experimentType",function(){_this6.model.classicURL.save({experimentType:_this6.model.state.get("experimentType")})},this),this.collection.activeExperiments.on("destroy",function(){_this6.fetchListCollection()},this),this.children.caption=new _TableCaption2.default({model:{state:this.model.state,application:this.model.application,uiPrefs:this.model.uiPrefs,user:_sharedmodels2.default.get("user"),serverInfo:_sharedmodels2.default.get("serverInfo"),rawSearch:this.rawSearch},collection:{lookupModels:this.collection.activeExperiments},noFilterButtons:!0,filterKey:captionFilterKey,countLabel:"Experiments",inputPlaceholder:"Filter by experiment name"}),this.children.caption.render().appendTo(this.experimentsList$El),this.children.modelsView=new _Master2.default({model:{state:this.model.state,application:this.model.application,uiPrefs:this.model.uiPrefs,userPref:_sharedmodels2.default.get("userPref"),user:_sharedmodels2.default.get("user"),appLocal:_sharedmodels2.default.get("app"),serverInfo:_sharedmodels2.default.get("serverInfo")},collection:{lookupModels:this.collection.activeExperiments,roles:_sharedmodels2.default.get("roles"),apps:_sharedmodels2.default.get("appLocals")}}),this.children.modelsView.startListening(),this.children.modelsView.render().appendTo(this.experimentsList$El),this.model.state.set("experimentPageRenderred",!0),this.manageStateOfChildren(),this},template:'\n        <div id="'+experimentsListContainerId+'"></div>'},{SORTED_EXPERIMENT_TYPES:[_ExperimentModel2.default.TYPES.PREDICT_NUMERIC_FIELDS,_ExperimentModel2.default.TYPES.PREDICT_CATEGORICAL_FIELDS,_ExperimentModel2.default.TYPES.DETECT_NUMERIC_OUTLIERS,_ExperimentModel2.default.TYPES.DETECT_CATEGORICAL_OUTLIERS,_ExperimentModel2.default.TYPES.FORECAST_TIME_SERIES,_ExperimentModel2.default.TYPES.CLUSTER_NUMERIC_EVENTS]});new ExperimentsView}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},567:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(513),__webpack_require__("shim/jquery"),__webpack_require__("views/shared/Modal"),__webpack_require__("views/shared/FlashMessages"),__webpack_require__("views/shared/controls/TextControl"),__webpack_require__("views/shared/controls/TextareaControl"),__webpack_require__("views/shared/controls/ControlGroup"),__webpack_require__(535),__webpack_require__(510),__webpack_require__("splunkjs/mvc/utils"),__webpack_require__("uri/route")],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_underscoreMltk,_jquery,_Modal,_FlashMessages,_TextControl,_TextareaControl,_ControlGroup,_ExperimentModel,_ShowcaseInfo,_utils,_route){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _underscoreMltk2=_interopRequireDefault(_underscoreMltk),_jquery2=_interopRequireDefault(_jquery),_Modal2=_interopRequireDefault(_Modal),_FlashMessages2=_interopRequireDefault(_FlashMessages),_TextControl2=_interopRequireDefault(_TextControl),_TextareaControl2=_interopRequireDefault(_TextareaControl),_ControlGroup2=_interopRequireDefault(_ControlGroup),_ExperimentModel2=_interopRequireDefault(_ExperimentModel),_ShowcaseInfo2=_interopRequireDefault(_ShowcaseInfo),_utils2=_interopRequireDefault(_utils),_route2=_interopRequireDefault(_route);exports.default=_Modal2.default.extend({className:_Modal2.default.CLASS_NAME+" "+_Modal2.default.CLASS_MODAL_WIDE,initialize:function(options){if(_Modal2.default.prototype.initialize.apply(this,arguments),this.model.inmem=this.model.experiment.clone(),this.model.inmem.entry.content.set({type:options.experimentType}),this.children.flashMessages=new _FlashMessages2.default({model:{experiment:this.model.inmem,experimentContent:this.model.inmem.entry.content}}),options.showExperimentTypePicker){var experimentTypeList=options.experimentTypes.map(function(type){return{label:_ShowcaseInfo2.default.assistants[_ExperimentModel2.default.TYPE_ASSISTANT_MAPPING[type]].title,value:type}});this.children.experimentTypePicker=new _ControlGroup2.default({className:"control-group",controlType:"SyntheticSelect",controlClass:"controls-block",controlOptions:{modelAttribute:"type",model:this.model.inmem.entry.content,items:experimentTypeList,toggleClassName:"btn",popdownOptions:{attachDialogTo:".modal:visible",scrollContainer:".modal:visible .modal-body:visible"}},label:(0,_underscoreMltk2.default)("Experiment Type").t()})}else this.children.experimentTypeLabel=new _ControlGroup2.default({controlType:"Label",controlOptions:{defaultValue:_ShowcaseInfo2.default.assistants[this.model.inmem.getAssistantPageName()].title},label:(0,_underscoreMltk2.default)("Experiment Type").t()});this.children.experimentTitleControl=new _TextControl2.default({elementId:"experiment-title-control",modelAttribute:"title",model:this.model.inmem.entry.content}),this.children.experimentTitle=new _ControlGroup2.default({controls:this.children.experimentTitleControl,controlType:"Text",controlClass:"controls-block",label:(0,_underscoreMltk2.default)("Experiment Title").t()}),this.children.descriptionControl=new _TextareaControl2.default({elementId:"experiment-description-control",modelAttribute:"description",model:this.model.inmem.entry.content,placeholder:"Optional"}),this.children.description=new _ControlGroup2.default({controls:this.children.descriptionControl,controlClass:"controls-block",label:(0,_underscoreMltk2.default)("Description").t()})},events:_jquery2.default.extend({},_Modal2.default.prototype.events,{"click a.modal-btn-primary":function(e){var _this=this;e.preventDefault(),this.model.inmem.app=this.model.application.get("app"),this.model.inmem.owner=this.model.application.get("owner"),this.model.inmem.save({},{data:{app:this.model.application.get("app"),owner:this.model.application.get("owner")},success:function(){var destinationAssistantPage=_this.model.inmem.getAssistantPageName(),pageInfo=_utils2.default.getPageInfo(),urlEncodedExperimentId={data:{experimentId:_this.model.inmem.getId()}};window.location=_route2.default.page(pageInfo.root,pageInfo.locale,pageInfo.app,destinationAssistantPage,urlEncodedExperimentId)}})}}),render:function(){return this.$el.html(_Modal2.default.TEMPLATE),this.$el.addClass("create-experiment-modal"),this.$(_Modal2.default.HEADER_TITLE_SELECTOR).text("Create New Experiment"),this.$(_Modal2.default.FOOTER_SELECTOR).append(_Modal2.default.BUTTON_CANCEL),this.$(_Modal2.default.FOOTER_SELECTOR).append((0,_jquery2.default)('<a href="#" class="btn btn-primary modal-btn-primary">').text("Create")),this.$(_Modal2.default.BODY_SELECTOR).append(_Modal2.default.FORM_HORIZONTAL_JUSTIFIED),this.children.flashMessages.render().appendTo(this.$(_Modal2.default.BODY_FORM_SELECTOR)),this.children.experimentTypePicker&&this.children.experimentTypePicker.render().appendTo(this.$(_Modal2.default.BODY_FORM_SELECTOR)),this.children.experimentTypeLabel&&this.children.experimentTypeLabel.render().appendTo(this.$(_Modal2.default.BODY_FORM_SELECTOR)),this.children.experimentTitle.render().appendTo(this.$(_Modal2.default.BODY_FORM_SELECTOR)),this.children.description.render().appendTo(this.$(_Modal2.default.BODY_FORM_SELECTOR)),this}}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},568:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__("require/backbone"),__webpack_require__(513),__webpack_require__(510),__webpack_require__(535)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_backbone,_underscoreMltk,_ShowcaseInfo,_ExperimentModel){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_backbone2=_interopRequireDefault(_backbone),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_ShowcaseInfo2=_interopRequireDefault(_ShowcaseInfo),_ExperimentModel2=_interopRequireDefault(_ExperimentModel);exports.default=_backbone2.default.View.extend({tagName:"ul",className:"mltk-experiments-type-filter",initialize:function(){var options=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},experimentTypes=options.experimentTypes||[];this.experimentTypes=experimentTypes.map(function(type){var assistantId=_ExperimentModel2.default.TYPE_ASSISTANT_MAPPING[type],assistant=_ShowcaseInfo2.default.assistants[assistantId];return{id:type,title:assistant.title,icon:assistant.icon}}),this.listenTo(this.model.stateModel,"change:experimentType",this.updateSelectedType),this.listenTo(this.model.countModel,"change",this.updateTypeCounts)},events:{"click li":function(e){var index=(0,_jquery2.default)(e.currentTarget).index(),experimentInfo=this.experimentTypes[index];this.model.stateModel.set("experimentType",experimentInfo.id),e.preventDefault()}},render:function(){var template=_underscoreMltk2.default.template(this.template,{experimentTypes:this.experimentTypes});return this.$el.html(template),this.updateSelectedType(),this.updateTypeCounts(),this},updateSelectedType:function(){var _this=this,newType=this.model.stateModel.get("experimentType");this.$el.find("> li > div.mltk-experiment-type-card").each(function(i,el){(0,_jquery2.default)(el)[newType===_this.experimentTypes[i].id?"addClass":"removeClass"]("active")})},updateTypeCounts:function(){var _this2=this;this.$el.find("> li").each(function(i,el){(0,_jquery2.default)(el).find(".count").text(_this2.model.countModel.get(_this2.experimentTypes[i].id))})},template:'\n        <% experimentTypes.forEach(function(type) { %>\n            <li>\n                <div class="mltk-experiment-type-card">\n                    <div class="mltk-experiment-type-title"><%= type.title %></div>\n                    <div class="mltk-experiment-type-info">\n                        <span class="icon <%= type.icon %>"></span>\n                        <span class="count"></span>\n                    </div>\n                <div>\n            </li>\n        <% }) %>\n    '}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},570:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(513),__webpack_require__(571),__webpack_require__(577),__webpack_require__(580),__webpack_require__(602)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_underscoreMltk,_Master,_Details,_TableRow,_MoreInfo){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _underscoreMltk2=_interopRequireDefault(_underscoreMltk),_Master2=_interopRequireDefault(_Master),_Details2=_interopRequireDefault(_Details),_TableRow2=_interopRequireDefault(_TableRow),_MoreInfo2=_interopRequireDefault(_MoreInfo),nameOptions={nameLabel:"Experiment Name",nameKey:"title"};exports.default=_Master2.default.extend({className:"mltk-experiments-table",details:{view:_Details2.default,options:_underscoreMltk2.default.extend({},nameOptions)},tableRow:{view:_TableRow2.default,options:_underscoreMltk2.default.extend({hasActions:!0},nameOptions)},moreInfo:{view:_MoreInfo2.default},tbodyClass:"mltk-experiments-listings",initialize:function(){this.columns=[{label:"i",className:"col-info",html:'<i class="icon-info"></i>'},{label:nameOptions.nameLabel,sortKey:nameOptions.nameKey},{label:"Algorithm",className:"col-algorithm"},{label:"i",className:"col-scheduled-training",html:'<i class="icon-large icon-clock"></i>'},{label:"i",className:"col-alert",html:'<i class="icon-large icon-bell"></i>'},{label:"Actions",className:"col-actions",visible:function(){return this.tableRow.options.hasActions}.bind(this)}],_Master2.default.prototype.initialize.apply(this,arguments)},updateTable:function(){var canDelete=this.collection.lookupModels.some(function(model){return model.canDelete()});this.tableRow.options.hasActions=0===this.collection.lookupModels.length||canDelete,this.children.head.render(),_Master2.default.prototype.updateTable.apply(this,arguments)}}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},577:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__(513),__webpack_require__("collections/Base"),__webpack_require__(572),__webpack_require__(574),__webpack_require__(578)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_underscoreMltk,_Base,_Details,_PermissionsDialog,_Master){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_Base2=_interopRequireDefault(_Base),_Details2=_interopRequireDefault(_Details),_PermissionsDialog2=_interopRequireDefault(_PermissionsDialog),_Master2=_interopRequireDefault(_Master);exports.default=_Details2.default.extend({events:{"click a.edit-permissions":function(e){this.children.permissionsDialog=new _PermissionsDialog2.default({document:this.model.lookupsModel,nameModel:this.model.lookupsModel.experimentInfo,user:this.model.user,serverInfo:this.model.serverInfo,application:this.model.application},this.collection,this.options),this.children.permissionsDialog.render().appendTo((0,_jquery2.default)("body")).show(),e.preventDefault()}},render:function(){var preprocessingList=this.model.lookupsModel.getPreprocessingSearchStageModels().map(function(stage){return{algorithm:stage.get("algorithm"),modelName:stage.get("modelName")}}).filter(function(stage){return null!=stage.algorithm&&stage.algorithm.length>0}),experimentSettings=new _Master2.default({title:"EXPERIMENT SETTINGS",collection:{searchStages:new _Base2.default(this.model.lookupsModel.getMainSearchStageModel())}}).render();return _underscoreMltk2.default.extend(this.innerTemplateParams,{preprocessingSteps:preprocessingList,dataSource:this.model.lookupsModel.dataSource.getFormattedString()}),this.innerTemplate='\n            <dt class="mltk-data-source">Dataset</dt>\n            <dd class="mltk-data-source"><%- dataSource %></dd>\n            <dt class="modified">Modified</dt>\n            <dd class="modified"></dd>\n            <% if (preprocessingSteps.length > 0) { %>\n                <br />\n                <h5>PREPROCESSING STEPS</h5>\n                <% preprocessingSteps.forEach(function(step) { %>\n                    <dt class="mltk-algorithm">Algorithm</dt>\n                    <dd class="mltk-algorithm"><%- step.algorithm %></dd>\n                    <% if (step.modelName) { %>\n                        <dt class="mltk-model-name">Model Name</dt>\n                        <dd class="mltk-model-name"><%- step.modelName %></dd>\n                    <% } %>\n                <% }) %>\n            <% } %>\n        ',_Details2.default.prototype.render.apply(this,arguments),this.$el.append(experimentSettings.el).addClass("mltk-experiment-details"),this}}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},580:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__("views/Base"),__webpack_require__(574),__webpack_require__(576),__webpack_require__(581),__webpack_require__("uri/route"),__webpack_require__("splunkjs/mvc/utils")],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_Base,_PermissionsDialog,_TableRow,_Master,_route,_utils){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_Base2=_interopRequireDefault(_Base),_PermissionsDialog2=_interopRequireDefault(_PermissionsDialog),_TableRow2=_interopRequireDefault(_TableRow),_Master2=_interopRequireDefault(_Master),_route2=_interopRequireDefault(_route),_utils2=_interopRequireDefault(_utils),scheduledTrainingTdClass="has-scheduled-training",alertTdClass="has-alerts";exports.default=_TableRow2.default.extend({className:"expand",initialize:function(){_Base2.default.prototype.initialize.apply(this,arguments),this.$el.addClass(this.options.index%2?"even":"odd"),this.children.manageDropdownView=new _Master2.default({model:{application:this.model.application,experiment:this.model.lookupsModel,user:this.model.user,appLocal:this.model.appLocal,serverInfo:this.model.serverInfo,state:this.model.state},collection:{roles:this.collection.roles},scheduledTrainingEnabled:!0,titleDescriptionEnabled:!0,createAlertEnabled:!0,button:!1})},startListening:function(){var _this=this;this.listenTo(this.model.lookupsModel,"updateCollection",function(){_this.model.state.trigger("change:search")}),this.listenTo(this.model.lookupsModel,"scheduleSuccess",this.updateScheduledTraining),this.listenTo(this.model.lookupsModel,"alertSuccess",this.updateAlert)},updateScheduledTraining:function(){var shouldBeActive=this.model.lookupsModel.hasSchedule(),clockIcon=this.$("td."+scheduledTrainingTdClass+" i");clockIcon.toggleClass("active-icon",shouldBeActive);var tooltipMessage=shouldBeActive?"Scheduled training":"No scheduled training";clockIcon.tooltip("destroy"),clockIcon.tooltip({title:tooltipMessage,animatation:!1,container:"body"})},updateAlert:function(){var shouldBeActive=this.model.lookupsModel.hasEnabledAlerts(),bellIcon=this.$("td."+alertTdClass+" i");bellIcon.toggleClass("active-icon",shouldBeActive);var tooltipMessage=shouldBeActive?"Active alerts":"No active alerts";bellIcon.tooltip("destroy"),bellIcon.tooltip({title:tooltipMessage,animatation:!1,container:"body"})},events:{"click a.edit-permissions":function(e){this.children.permissionsDialog=new _PermissionsDialog2.default({document:this.model.lookupsModel,nameModel:this.model.lookupsModel.experimentInfo,user:this.model.user,serverInfo:this.model.serverInfo,application:this.model.application},this.collection,this.options),this.children.permissionsDialog.render().appendTo((0,_jquery2.default)("body")).show(),e.preventDefault()}},render:function(){var destinationAssistantPage=this.model.lookupsModel.getAssistantPageName(),pageInfo=_utils2.default.getPageInfo(),urlEncodedExperimentId={data:{experimentId:this.model.lookupsModel.getId()}};return this.$el.html(this.compiledTemplate({experimentName:this.model.lookupsModel.getFormattedName(),experimentLink:_route2.default.page(pageInfo.root,pageInfo.locale,pageInfo.app,destinationAssistantPage,urlEncodedExperimentId),hasActions:this.options.hasActions,algorithm:this.model.lookupsModel.getAlgorithm()})),this.children.manageDropdownView.render().prependTo(this.$(".actions-edit")),this.updateScheduledTraining(),this.updateAlert(),this},template:'\n        <td class="expands">\n            <a href="#"><i class="icon-triangle-right-small"></i></a>\n        </td>\n        <td class="title">\n            <a href="<%= experimentLink %>" title="<%- experimentName %>" class=""><%- experimentName %></a>\n        </td>\n        <td class="algorithm">\n            <%- algorithm %>\n        </td>\n        <td class="'+scheduledTrainingTdClass+' icon-cell">\n            <i class="icon-large icon-clock"></i>\n        </td>\n        <td class="'+alertTdClass+' icon-cell">\n            <i class="icon-large icon-bell"></i>\n        </td>\n        <% if (hasActions) { %>\n        <td class="actions actions-edit">\n        </td>\n        <% } %>\n    '}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},602:function(module,exports,__webpack_require__){
var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__(583),__webpack_require__(575)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_TitleDescription,_MoreInfo){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_TitleDescription2=_interopRequireDefault(_TitleDescription),_MoreInfo2=_interopRequireDefault(_MoreInfo);exports.default=_MoreInfo2.default.extend({events:{"click a.edit-description":function(e){this.children.titleDescriptionModal=new _TitleDescription2.default({model:{experiment:this.model.lookupsModel},onHiddenRemove:!0}),this.children.titleDescriptionModal.render().appendTo((0,_jquery2.default)("body")).show(),e.preventDefault()}},render:function(){_MoreInfo2.default.prototype.render.apply(this,arguments);var canWrite=this.model.lookupsModel.canWrite();return canWrite&&this.$("p.description").append('<a class="edit-description" href="#">Edit</a>'),this}}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},603:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__("views/Base"),__webpack_require__(513),__webpack_require__(535),__webpack_require__(510),__webpack_require__(567)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_Base,_underscoreMltk,_ExperimentModel,_ShowcaseInfo,_Create){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function _defineProperty(obj,key,value){return key in obj?Object.defineProperty(obj,key,{value:value,enumerable:!0,configurable:!0,writable:!0}):obj[key]=value,obj}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_Base2=_interopRequireDefault(_Base),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_ExperimentModel2=_interopRequireDefault(_ExperimentModel),_ShowcaseInfo2=_interopRequireDefault(_ShowcaseInfo),_Create2=_interopRequireDefault(_Create),experimentTypeListClass="mltk-experiments-type-list";exports.default=_Base2.default.extend({className:"mltk-experiments-type-selector",initialize:function(){var options=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},experimentTypes=options.experimentTypes||[];this.experimentTypes=experimentTypes.map(function(type){var assistantId=_ExperimentModel2.default.TYPE_ASSISTANT_MAPPING[type],assistant=_ShowcaseInfo2.default.assistants[assistantId];return{id:assistantId,alternateId:assistant.alternateId,title:assistant.title,icon:assistant.icon,description:assistant.description,examples:assistant.experimentExamples.map(function(exampleId){return assistant.examples[exampleId]})}})},events:_defineProperty({},"click ."+experimentTypeListClass+" > li",function(e){var _this=this,index=(0,_jquery2.default)(e.currentTarget).index(),experimentInfo=this.experimentTypes[index];this.model.experiment=new _ExperimentModel2.default,this.model.experiment.app=this.model.application.get("app"),this.model.experiment.owner=this.model.application.get("owner");var experimentDeferred=this.model.experiment.fetch();_jquery2.default.when(experimentDeferred).always(function(){var createExperimentModal=new _Create2.default({experimentType:experimentInfo.alternateId,model:{experiment:_this.model.experiment,application:_this.model.application},onHiddenRemove:!0});createExperimentModal.render().appendTo((0,_jquery2.default)("body")).show()}),e.preventDefault()}),render:function(){var template=_underscoreMltk2.default.template(this.template,{experimentTypes:this.experimentTypes});return this.$el.html(template),this},template:'\n        <h2>Select an Assistant to Create an Experiment</h2>\n        <ul class="'+experimentTypeListClass+'">\n            <% experimentTypes.forEach(function(experimentType) { %>\n                <li>\n                    <div class="'+experimentTypeListClass+'-title">\n                        <span class="icon <%= experimentType.icon %>"></span>\n                        <h3><%= experimentType.title %></h3>\n                    </div>\n                    <%= experimentType.description %> For example,\n                    <ul class="'+experimentTypeListClass+'-examples">\n                        <% experimentType.examples.forEach(function(example) { %>\n                            <li><%= example.label %></li>\n                        <% }) %>\n                    </ul>\n                </li>\n            <% }) %>\n        </ul>\n    '}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},604:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__("collections/SplunkDsBase"),__webpack_require__(535)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_SplunkDsBase,_ExperimentModel){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_SplunkDsBase2=_interopRequireDefault(_SplunkDsBase),_ExperimentModel2=_interopRequireDefault(_ExperimentModel);exports.default=_SplunkDsBase2.default.extend({model:_ExperimentModel2.default,url:"mltk/experiments",initialize:function(){var options=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};_SplunkDsBase2.default.prototype.initialize.apply(this,arguments),this.experimentType=options.experimentType},fetch:function(fetchParams){var clonedParams=_jquery2.default.extend(!0,{},fetchParams);null!=this.experimentType&&(clonedParams.data.search?clonedParams.data.search="("+clonedParams.data.search+") AND (type="+this.experimentType+")":clonedParams.data.search="type="+this.experimentType),_SplunkDsBase2.default.prototype.fetch.call(this,clonedParams)}}),module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}});