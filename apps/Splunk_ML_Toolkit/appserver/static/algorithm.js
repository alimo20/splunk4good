webpackJsonp([8],{0:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__webpack_require__.p=function(){function make_url(){for(var seg,len,output="",i=0,l=arguments.length;i<l;i++)seg=arguments[i].toString(),len=seg.length,len>1&&"/"==seg.charAt(len-1)&&(seg=seg.substring(0,len-1)),output+="/"!=seg.charAt(0)?"/"+seg:seg;if("/"!=output){var segments=output.split("/"),firstseg=segments[1];if("static"==firstseg||"modules"==firstseg){var postfix=output.substring(firstseg.length+2,output.length);output="/"+firstseg,window.$C.BUILD_NUMBER&&(output+="/@"+window.$C.BUILD_NUMBER),window.$C.BUILD_PUSH_NUMBER&&(output+="."+window.$C.BUILD_PUSH_NUMBER),"app"==segments[2]&&(output+=":"+getConfigValue("APP_BUILD",0)),output+="/"+postfix}}var root=getConfigValue("MRSPARKLE_ROOT_PATH","/"),djangoRoot=getConfigValue("DJANGO_ROOT_PATH",""),locale=getConfigValue("LOCALE","en-US"),combinedPath="";return combinedPath=djangoRoot&&output.substring(0,djangoRoot.length)===djangoRoot?output.replace(djangoRoot,djangoRoot+"/"+locale.toLowerCase()):"/"+locale+output,""==root||"/"==root?combinedPath:root+combinedPath}function getConfigValue(key,defaultValue){if(window.$C&&window.$C.hasOwnProperty(key))return window.$C[key];if(void 0!==defaultValue)return defaultValue;throw new Error("getConfigValue - "+key+" not set, no default provided")}return make_url("/static/app/Splunk_ML_Toolkit/")+"/"}(),__WEBPACK_AMD_DEFINE_ARRAY__=[__webpack_require__(694),__webpack_require__("util/router_utils"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(_Algorithm,_router_utils){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}var _Algorithm2=_interopRequireDefault(_Algorithm),_router_utils2=_interopRequireDefault(_router_utils);new _Algorithm2.default,_router_utils2.default.start_backbone_history()}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},694:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(695),__webpack_require__(696),__webpack_require__("algorithm/Master"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_underscoreMltk,_Base,_Master){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _underscoreMltk2=_interopRequireDefault(_underscoreMltk),_Base2=_interopRequireDefault(_Base),_Master2=_interopRequireDefault(_Master),AlgorithmRouter=_Base2.default.extend({initialize:function(){_Base2.default.prototype.initialize.apply(this,arguments),this.setPageTitle((0,_underscoreMltk2.default)("Algorithm Settings").t())},page:function(){var _this=this;_Base2.default.prototype.page.apply(this,arguments),this.mlsplView&&this.mlsplView.remove(),this.deferreds.user.then(function(){_this.mlsplView=new _Master2.default({model:{application:_this.model.application,classicurl:_this.model.classicurl,user:_this.model.user},deferreds:{layout:_this.deferreds.layout}})})}});exports.default=AlgorithmRouter,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"algorithm/Master":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;(function(module){__WEBPACK_AMD_DEFINE_ARRAY__=[exports,__webpack_require__("shim/jquery"),module,__webpack_require__("util/splunkd_utils"),__webpack_require__("views/shared/FlashMessages"),__webpack_require__(697),__webpack_require__("shared/BaseDashboard"),__webpack_require__(731),__webpack_require__("algorithm/mlspl/Master"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(exports,_jquery,_module2,_splunkd_utils,_FlashMessages,_MLSPL,_BaseDashboard,_url,_Master){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_module3=_interopRequireDefault(_module2),_splunkd_utils2=_interopRequireDefault(_splunkd_utils),_FlashMessages2=_interopRequireDefault(_FlashMessages),_MLSPL2=_interopRequireDefault(_MLSPL),_BaseDashboard2=_interopRequireDefault(_BaseDashboard),_Master2=_interopRequireDefault(_Master),syncOptions={data:{app:"Splunk_ML_Toolkit",owner:"nobody"},emulateJSON:!0},AlgorithmView=_BaseDashboard2.default.extend({moduleId:_module3.default.id,headerOptions:{title:"Algorithm Settings"},initialize:function(options){var _this=this;this.deferreds.mlspl=_jquery2.default.Deferred(),this.requiredDeferredIds.push("mlspl"),_BaseDashboard2.default.prototype.initialize.call(this,options);var stanza=this.model.classicurl.get("stanza");null!=stanza&&""!==stanza||this.deferreds.mlspl.reject("No algorithm was specified."),this.model.mlspl=new _MLSPL2.default({id:stanza}),this.model.mlspl.bootstrap(syncOptions,this.model.user.isAdminLike()).done(function(model,response){_this.deferreds.mlspl.resolve()}).fail(function(model,response){_this.deferreds.mlspl.reject('Unable to load settings for "'+_this.model.mlspl.id+'".')})},returnToAlgorithmList:function(){window.location=(0,_url.buildMLTKPageUrl)(this.model.application,"settings")},render:function(){_BaseDashboard2.default.prototype.render.call(this);var algorithm=this.model.classicurl.get("stanza");"default"===algorithm?this.model.header.set({title:"Default Algorithm Settings",description:"Configure default settings for the fit and apply commands here.\n                              All algorithms will use these settings unless specifically configured with their own settings."}):this.model.header.set({title:algorithm+" Algorithm",description:"Configure settings for the fit and apply commands for the "+algorithm+" algorithm here.\n                              Any settings not configured on the algorithm directly will be inherited from the default settings."}),this.children.header.render(),this.children.flashMessage=new _FlashMessages2.default({model:{mlspl:this.model.mlspl}}),this.model.mlspl.off("validated",null,this.children.flashMessage.flashMsgHelper);var isAdminLike=this.model.user.isAdminLike();return isAdminLike||this.children.flashMessage.flashMsgHelper.addGeneralMessage("notAdminLike",{type:_splunkd_utils2.default.WARNING,html:"You do not have permissions to edit this configuration."}),this.children.formView=new _Master2.default({model:{mlspl:this.model.mlspl},labelWidth:300,readOnly:!isAdminLike,onSave:function(){var _this2=this;this.children.formView.model.props.set("disabled",!0),this.model.mlspl.save(null,syncOptions).done(function(){_this2.returnToAlgorithmList()}).always(function(){_this2.children.formView.model.props.set("disabled",!1)})}.bind(this),onCancel:this.returnToAlgorithmList.bind(this)}),this.$el.append(this.children.flashMessage.render().el,this.children.formView.render().el),this}});exports.default=AlgorithmView,_module3.default.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}).call(exports,__webpack_require__(8)(module))},697:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(698),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_Properties){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _Properties2=_interopRequireDefault(_Properties),validation={handle_new_cat:{oneOf:["default","skip","stop"]},max_distinct_cat_values:{pattern:"digits"},max_distinct_cat_values_for_classifiers:{pattern:"digits"},max_distinct_cat_values_for_scoring:{pattern:"digits"},max_fit_time:{pattern:"digits"},max_inputs:{pattern:"digits"},max_memory_usage_mb:{pattern:"digits"},max_model_size_mb:{pattern:"digits"},max_score_time:{pattern:"digits"},streaming_apply:{oneOf:["true","false"]},summary_depth_limit:{pattern:"digits",required:!1},summary_return_json:{oneOf:["true","false"],required:!1},use_sampling:{oneOf:["true","false"]}};Object.keys(validation).forEach(function(key){null!=validation[key].pattern?"digits"===validation[key].pattern&&(validation[key].msg=key+" must only contain digits"):null!=validation[key].oneOf&&(validation[key].msg=key+" must be one of: "+validation[key].oneOf.join(", "))});var MLSPLPropertiesModel=_Properties2.default.extend({file:"mlspl",description:{handle_new_cat:"\n            Action to perform when new value(s) for categorical variable/explanatory variable is encountered in partial_fit.\n            - default: set all values of the column that corresponds to the new categorical value to 0's\n            - skip: skip over rows that contain the new value(s) and raise a warning\n            - stop: stop the operation by raising an error\n        ",max_distinct_cat_values:"\n            Determines the upper limit for the number of categorical values that will be encoded in one-hot encoding.\n            If the number of distinct values exceeds this limit, the field will be dropped (with a warning).\n        ",max_distinct_cat_values_for_classifiers:"\n            Determines the upper limit for the number of distinct values in a categorical field that is the target (or response) variable in a classifier algorithm.\n            If the number of distinct values exceeds this limit, the field will be dropped (with a warning).\n        ",max_distinct_cat_values_for_scoring:"\n            Determines the upper limit for the number of distinct values in a categorical field that is the target (or response) variable in a scoring method.\n            If the number of distinct values exceeds this limit, the field will be dropped (with an appropriate warning or error message).\n        ",max_fit_time:'The maximum time, in seconds, to spend in the "fit" phase of an algorithm.',max_inputs:'\n            The maximum number of events an algorithm considers when fitting a model.\n            If this limit is exceeded, follows the behavior defined by "use_sampling".\n        ',max_memory_usage_mb:"The maximum allowed memory usage, in megabytes, by the fit command while fitting a model.",max_model_size_mb:"The maximum allowed size of a model, in megabytes, created by the fit command.",max_score_time:'The maximum time, in seconds, to spend in the "score" phase of an algorithm',streaming_apply:'Setting streaming_apply to true allows the execution of apply command at indexer level. Otherwise "apply" is done on search head.',summary_depth_limit:'The number of nodes in a decision tree to display when running the "summary" command on a model.',summary_return_json:'Whether or not to return a json representation instead of a ASCII representation of the nodes when running the "summary" command on a model.',use_sampling:"Indicates whether to use Reservoir Sampling for data sets that exceed max_inputs or to instead throw an error"},validation:validation});exports.default=MLSPLPropertiesModel,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},698:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__("shim/jquery"),__webpack_require__(695),__webpack_require__("util/splunkd_utils"),__webpack_require__("models/Base"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_jquery,_underscoreMltk,_splunkd_utils,_Base){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _jquery2=_interopRequireDefault(_jquery),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_splunkd_utils2=_interopRequireDefault(_splunkd_utils),_Base2=_interopRequireDefault(_Base),_extends=Object.assign||function(target){for(var i=1;i<arguments.length;i++){var source=arguments[i];for(var key in source)Object.prototype.hasOwnProperty.call(source,key)&&(target[key]=source[key])}return target},PropertiesModel=_Base2.default.extend({file:null,urlRoot:"properties",url:function url(){if(null!=this.file){var url=this.urlRoot+"/"+encodeURIComponent(this.file);return null!=this.id?url+"/"+encodeURIComponent(this.id):url}return this.urlRoot},bootstrap:function(){var syncOptions=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},_this=this,isAdminLike=arguments[1],bootstrapDeferred=arguments.length>2&&void 0!==arguments[2]?arguments[2]:_jquery2.default.Deferred(),proxyModel=new this.constructor({id:this.id});return proxyModel.fetch(_extends({success:function(){_this.set(proxyModel.toJSON()),bootstrapDeferred.resolve()},error:function(){if(isAdminLike){var createModel=new _this.constructor;createModel.save({__stanza:_this.id},_extends({dataType:"text"},syncOptions)).done(function(){proxyModel.fetch(_extends({success:function(){_this.set(proxyModel.toJSON()),bootstrapDeferred.resolve()},error:function(){bootstrapDeferred.reject()}},syncOptions))}).fail(function(){bootstrapDeferred.reject()})}else{var defaultModel=new _this.constructor({id:"default"});defaultModel.fetch(_extends({success:function(){var defaultJSON=defaultModel.toJSON();defaultJSON.id=_this.id,_this.set(defaultJSON),bootstrapDeferred.resolve()},error:function(){bootstrapDeferred.reject()}},syncOptions)).fail(function(){bootstrapDeferred.reject()})}}},syncOptions)),bootstrapDeferred},sync:function(method,model){var options=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{},url=_underscoreMltk2.default.isFunction(model.url)?model.url():model.url,defaults={data:{output_mode:"json"},url:_splunkd_utils2.default.fullpath(url,options.data)};_jquery2.default.extend(!0,defaults,options,{data:model.attributes}),delete defaults.data.app,delete defaults.data.owner,delete defaults.data.sharing,delete defaults.data.id;var newMethod="update"===method?"create":method;return _Base2.default.prototype.sync.apply(this,[newMethod,model,defaults])},parse:function(response){var data=(arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},{});return null!=response.entry&&response.entry.forEach(function(props){data[props.name]=props.content}),data}});exports.default=PropertiesModel,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"algorithm/mlspl/Master":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;(function(module){__WEBPACK_AMD_DEFINE_ARRAY__=[exports,__webpack_require__(392),__webpack_require__("require/backbone"),module,__webpack_require__("views/ReactAdapterBase"),__webpack_require__(697),__webpack_require__("algorithm/mlspl/FormContainer"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(exports,_react,_backbone,_module2,_ReactAdapterBase,_MLSPL,_FormContainer){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function _objectWithoutProperties(obj,keys){var target={};for(var i in obj)keys.indexOf(i)>=0||Object.prototype.hasOwnProperty.call(obj,i)&&(target[i]=obj[i]);return target}Object.defineProperty(exports,"__esModule",{value:!0});var _react2=_interopRequireDefault(_react),_backbone2=_interopRequireDefault(_backbone),_module3=_interopRequireDefault(_module2),_ReactAdapterBase2=_interopRequireDefault(_ReactAdapterBase),_MLSPL2=_interopRequireDefault(_MLSPL),_FormContainer2=_interopRequireDefault(_FormContainer);exports.default=_ReactAdapterBase2.default.extend({moduleId:_module3.default.id,initialize:function(){var options=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};_ReactAdapterBase2.default.prototype.initialize.apply(this,options),this.model=this.model||{};var props=(options.model,_objectWithoutProperties(options,["model"]));this.model.mlspl=this.model.mlspl||new _MLSPL2.default,this.model.props=this.model.props||new _backbone2.default.Model,this.model.props.set(props)},getComponent:function(){return _react2.default.createElement(_FormContainer2.default,{model:this.model})}}),_module3.default.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}).call(exports,__webpack_require__(8)(module))},"algorithm/mlspl/FormContainer":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(392),__webpack_require__(732),__webpack_require__("algorithm/mlspl/Form"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_react,_propTypes,_Form){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function _classCallCheck(instance,Constructor){if(!(instance instanceof Constructor))throw new TypeError("Cannot call a class as a function")}function _possibleConstructorReturn(self,call){if(!self)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!call||"object"!=typeof call&&"function"!=typeof call?self:call}function _inherits(subClass,superClass){if("function"!=typeof superClass&&null!==superClass)throw new TypeError("Super expression must either be null or a function, not "+typeof superClass);subClass.prototype=Object.create(superClass&&superClass.prototype,{constructor:{value:subClass,enumerable:!1,writable:!0,configurable:!0}}),superClass&&(Object.setPrototypeOf?Object.setPrototypeOf(subClass,superClass):subClass.__proto__=superClass)}Object.defineProperty(exports,"__esModule",{value:!0});var _react2=_interopRequireDefault(_react),_propTypes2=_interopRequireDefault(_propTypes),_Form2=_interopRequireDefault(_Form),_extends=Object.assign||function(target){for(var i=1;i<arguments.length;i++){var source=arguments[i];for(var key in source)Object.prototype.hasOwnProperty.call(source,key)&&(target[key]=source[key])}return target},_createClass=function(){function defineProperties(target,props){for(var i=0;i<props.length;i++){var descriptor=props[i];descriptor.enumerable=descriptor.enumerable||!1,descriptor.configurable=!0,"value"in descriptor&&(descriptor.writable=!0),Object.defineProperty(target,descriptor.key,descriptor)}}return function(Constructor,protoProps,staticProps){return protoProps&&defineProperties(Constructor.prototype,protoProps),staticProps&&defineProperties(Constructor,staticProps),Constructor}}(),propTypes={model:_propTypes2.default.shape({mlspl:_propTypes2.default.object.isRequired,props:_propTypes2.default.object.isRequired}).isRequired},MLSPLFormContainer=function(_Component){function MLSPLFormContainer(props,context){_classCallCheck(this,MLSPLFormContainer);var _this=_possibleConstructorReturn(this,(MLSPLFormContainer.__proto__||Object.getPrototypeOf(MLSPLFormContainer)).call(this,props,context));_this.handleValueChange=_this.handleValueChange.bind(_this);var _this$parseMLSPLModel=_this.parseMLSPLModelToProps(),attributes=_this$parseMLSPLModel.attributes,isValid=_this$parseMLSPLModel.isValid;return _this.state={attributes:attributes,isValid:isValid},_this}return _inherits(MLSPLFormContainer,_Component),_createClass(MLSPLFormContainer,[{key:"componentDidMount",value:function(){var _this2=this;this.props.model.props.on("change",function(model){_this2.setState(model.changed)}),this.props.model.mlspl.on("change",function(model){var _parseMLSPLModelToPro=_this2.parseMLSPLModelToProps(),attributes=_parseMLSPLModelToPro.attributes,isValid=_parseMLSPLModelToPro.isValid;_this2.setState({attributes:attributes,isValid:isValid})})}},{key:"parseMLSPLModelToProps",value:function(){var _this3=this,validation=this.props.model.mlspl.validate()||{},attributes=Object.keys(this.props.model.mlspl.toJSON()).sort().reduce(function(reduced,key){return"id"!==key&&reduced.push({error:validation[key],label:key,tooltip:_this3.props.model.mlspl.description[key],validation:_this3.props.model.mlspl.validation[key],value:_this3.props.model.mlspl.attributes[key]}),reduced},[]);return{attributes:attributes,isValid:0===Object.keys(validation).length,validation:validation}}},{key:"handleValueChange",value:function(value,key){this.props.model.mlspl.set(key,value)}},{key:"render",value:function(){return _react2.default.createElement(_Form2.default,_extends({onChange:this.handleValueChange},this.state,this.props.model.props.attributes))}}]),MLSPLFormContainer}(_react.Component);MLSPLFormContainer.propTypes=propTypes,exports.default=MLSPLFormContainer,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"algorithm/mlspl/Form":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(392),__webpack_require__(732),__webpack_require__(548),__webpack_require__(550),__webpack_require__(499),__webpack_require__(567),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_react,_propTypes,_ControlGroup,_Text,_Button,_RadioBar){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}function formatTooltip(){var tooltip=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"";return _react2.default.createElement("span",{style:{whiteSpace:"pre-line"}},tooltip.trim())}Object.defineProperty(exports,"__esModule",{value:!0});var _react2=_interopRequireDefault(_react),_propTypes2=_interopRequireDefault(_propTypes),_ControlGroup2=_interopRequireDefault(_ControlGroup),_Text2=_interopRequireDefault(_Text),_Button2=_interopRequireDefault(_Button),_RadioBar2=_interopRequireDefault(_RadioBar),_extends=Object.assign||function(target){for(var i=1;i<arguments.length;i++){var source=arguments[i];for(var key in source)Object.prototype.hasOwnProperty.call(source,key)&&(target[key]=source[key])}return target},propTypes={attributes:_propTypes2.default.arrayOf(_propTypes2.default.object),dataTest:_propTypes2.default.string,disabled:_propTypes2.default.bool,isValid:_propTypes2.default.bool,labelWidth:_propTypes2.default.number,onCancel:_propTypes2.default.func.isRequired,onChange:_propTypes2.default.func.isRequired,onSave:_propTypes2.default.func.isRequired,readOnly:_propTypes2.default.bool},defaultProps={attributes:[],dataTest:null,disabled:!1,readOnly:!1,isValid:!1,labelWidth:120},ConfList=function(_ref){var attributes=_ref.attributes,dataTest=_ref.dataTest,disabled=_ref.disabled,isValid=_ref.isValid,labelWidth=_ref.labelWidth,onCancel=_ref.onCancel,_onChange=_ref.onChange,onSave=_ref.onSave,readOnly=_ref.readOnly,extraProps={};return null!=dataTest&&(extraProps["data-test"]=dataTest),_react2.default.createElement(_react2.default.Fragment,null,attributes.map(function(attr){return _react2.default.createElement(_ControlGroup2.default,_extends({error:null!=attr.error,help:attr.error,key:attr.label,label:attr.label,labelWidth:labelWidth,tooltip:formatTooltip(attr.tooltip)},extraProps),attr.validation&&null!=attr.validation.oneOf?_react2.default.createElement(_RadioBar2.default,{error:null!=attr.error,name:attr.label,onChange:function(e,_ref2){var value=_ref2.value,name=_ref2.name;return _onChange(value,name)},value:attr.value},attr.validation.oneOf.map(function(option){return _react2.default.createElement(_RadioBar2.default.Option,{disabled:disabled||readOnly,key:option,label:option,value:option})})):_react2.default.createElement(_Text2.default,{disabled:disabled||readOnly,error:null!=attr.error,name:attr.label,onChange:function(e,_ref3){var value=_ref3.value,name=_ref3.name;return _onChange(value,name)},value:attr.value}))}),_react2.default.createElement(_ControlGroup2.default,{controlsLayout:"none",label:"",labelWidth:labelWidth},_react2.default.createElement(_Button2.default,{appearance:"secondary",disabled:disabled,label:"Cancel",onClick:onCancel}),!readOnly&&_react2.default.createElement(_Button2.default,{appearance:"primary",disabled:disabled||!isValid,label:"Save",onClick:onSave})))};ConfList.propTypes=propTypes,ConfList.defaultProps=defaultProps,exports.default=ConfList,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}});