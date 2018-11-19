webpackJsonp([10],{0:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__webpack_require__.p=function(){function make_url(){for(var seg,len,output="",i=0,l=arguments.length;i<l;i++)seg=arguments[i].toString(),len=seg.length,len>1&&"/"==seg.charAt(len-1)&&(seg=seg.substring(0,len-1)),output+="/"!=seg.charAt(0)?"/"+seg:seg;if("/"!=output){var segments=output.split("/"),firstseg=segments[1];if("static"==firstseg||"modules"==firstseg){var postfix=output.substring(firstseg.length+2,output.length);output="/"+firstseg,window.$C.BUILD_NUMBER&&(output+="/@"+window.$C.BUILD_NUMBER),window.$C.BUILD_PUSH_NUMBER&&(output+="."+window.$C.BUILD_PUSH_NUMBER),"app"==segments[2]&&(output+=":"+getConfigValue("APP_BUILD",0)),output+="/"+postfix}}var root=getConfigValue("MRSPARKLE_ROOT_PATH","/"),djangoRoot=getConfigValue("DJANGO_ROOT_PATH",""),locale=getConfigValue("LOCALE","en-US"),combinedPath="";return combinedPath=djangoRoot&&output.substring(0,djangoRoot.length)===djangoRoot?output.replace(djangoRoot,djangoRoot+"/"+locale.toLowerCase()):"/"+locale+output,""==root||"/"==root?combinedPath:root+combinedPath}function getConfigValue(key,defaultValue){if(window.$C&&window.$C.hasOwnProperty(key))return window.$C[key];if(void 0!==defaultValue)return defaultValue;throw new Error("getConfigValue - "+key+" not set, no default provided")}return make_url("/static/app/Splunk_ML_Toolkit/")+"/"}(),__WEBPACK_AMD_DEFINE_ARRAY__=[__webpack_require__(829),__webpack_require__("util/router_utils"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(_Contents,_router_utils){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}var _Contents2=_interopRequireDefault(_Contents),_router_utils2=_interopRequireDefault(_router_utils);new _Contents2.default,_router_utils2.default.start_backbone_history()}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},829:function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[module,exports,__webpack_require__(695),__webpack_require__(696),__webpack_require__("contents/Master"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(module,exports,_underscoreMltk,_Base,_Master){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _underscoreMltk2=_interopRequireDefault(_underscoreMltk),_Base2=_interopRequireDefault(_Base),_Master2=_interopRequireDefault(_Master),ContentsRouter=_Base2.default.extend({initialize:function(){_Base2.default.prototype.initialize.apply(this,arguments),this.setPageTitle((0,_underscoreMltk2.default)("Showcase").t())},page:function(){_Base2.default.prototype.page.apply(this,arguments),this.contentsView&&this.contentsView.remove(),this.contentsView=new _Master2.default({model:{classicurl:this.model.classicurl},deferreds:{layout:this.deferreds.layout}})}});exports.default=ContentsRouter,module.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"contents/Master":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;(function(module){__WEBPACK_AMD_DEFINE_ARRAY__=[exports,__webpack_require__("require/backbone"),module,__webpack_require__("splunkjs/mvc/dropdownview"),__webpack_require__("splunkjs/mvc/utils"),__webpack_require__("shim/splunk.util"),__webpack_require__(726),__webpack_require__(728),__webpack_require__("shared/BaseDashboard"),__webpack_require__("contents/List"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(exports,_backbone,_module2,_dropdownview,_utils,_splunk,_roleStorage,_showcaseInfo,_BaseDashboard,_List){"use strict";function _interopRequireWildcard(obj){if(obj&&obj.__esModule)return obj;var newObj={};if(null!=obj)for(var key in obj)Object.prototype.hasOwnProperty.call(obj,key)&&(newObj[key]=obj[key]);return newObj.default=obj,newObj}function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _backbone2=_interopRequireDefault(_backbone),_module3=_interopRequireDefault(_module2),_dropdownview2=_interopRequireDefault(_dropdownview),_utils2=_interopRequireDefault(_utils),_splunk2=_interopRequireDefault(_splunk),RoleStorage=_interopRequireWildcard(_roleStorage),_showcaseInfo2=_interopRequireDefault(_showcaseInfo),_BaseDashboard2=_interopRequireDefault(_BaseDashboard),_List2=_interopRequireDefault(_List),ContentsView=_BaseDashboard2.default.extend({moduleId:_module3.default.id,headerOptions:{title:"Showcase",description:'Welcome to the Showcase, which exhibits some of the analytics enabled by this app.\n                      Click on one of the examples to see that Assistant applied to a real dataset.\n                      Please see the <a href="http://tiny.cc/splunkmlvideos" class="external" target="_blank">video tutorials</a> for more information.'},initialize:function(options){_BaseDashboard2.default.prototype.initialize.call(this,options),this.model.showcaseInfo=new _backbone2.default.Model({summaries:[]}),this.showcaseListView=new _List2.default({model:this.model.showcaseInfo})},render:function(){var _this=this;return _BaseDashboard2.default.prototype.render.call(this),function(){var choices=Object.keys(_showcaseInfo2.default.roles).map(function(roleName){return{label:_showcaseInfo2.default.roles[roleName].label,value:roleName}}),rolePickerControl=new _dropdownview2.default({id:"rolePickerControl",el:_this.$el.find("#rolePickerControl"),labelField:"label",valueField:"value",showClearButton:!1,choices:choices}).render();return rolePickerControl.on("change",function(roleId){var role=_showcaseInfo2.default.roles[roleId],showcaseList=role.showcases.map(function(showcaseId){var assistantData=_showcaseInfo2.default.assistants[showcaseId],showcaseData=assistantData.showcases[roleId],imageFileName=(null!=assistantData.image?assistantData.image:showcaseId)+".png";return{dashboard:showcaseId,name:assistantData.title,description:assistantData.description+" "+showcaseData.description,imageURL:_splunk2.default.make_url("/static/app/"+_utils2.default.getPageInfo().app+"/img/content_thumbnails/"+imageFileName),examples:showcaseData.examples.map(function(exampleId){return assistantData.examples[exampleId]})}});_this.model.showcaseInfo.set("summaries",showcaseList),RoleStorage.setRole(roleId)}),rolePickerControl.val(RoleStorage.getRole()),rolePickerControl}(),this.$el.append(this.showcaseListView.render().el),this},template:'\n        <div class="input input-dropdown splunk-view">\n            <label>Select which examples to show</label>\n            <div id="rolePickerControl"></div>\n        </div>\n    '});exports.default=ContentsView,_module3.default.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}).call(exports,__webpack_require__(8)(module))},"contents/List":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;(function(module){__WEBPACK_AMD_DEFINE_ARRAY__=[exports,__webpack_require__("require/backbone"),__webpack_require__(695),module,__webpack_require__("contents/List.pcssm"),__webpack_require__(36)],__WEBPACK_AMD_DEFINE_RESULT__=function(exports,_backbone,_underscoreMltk,_module2,_List){"use strict";function _interopRequireDefault(obj){return obj&&obj.__esModule?obj:{default:obj}}Object.defineProperty(exports,"__esModule",{value:!0});var _backbone2=_interopRequireDefault(_backbone),_underscoreMltk2=_interopRequireDefault(_underscoreMltk),_module3=_interopRequireDefault(_module2),_List2=_interopRequireDefault(_List);exports.default=_backbone2.default.View.extend({moduleId:_module3.default.id,initialize:function(){this.listenTo(this.model,"change",this.render)},render:function(){var summaryData=this.model.get("summaries"),template=_underscoreMltk2.default.template(this.template,{data:summaryData});return this.$el.html(template),this},template:'\n        <ul class="'+_List2.default.list+'">\n            <% data.forEach(function(summary) { %>\n                <li class="'+_List2.default.listItem+'">\n                    <img class="'+_List2.default.image+'" src="<%- summary.imageURL %>">\n                    <div class="'+_List2.default.content+'" data-test="showcase-list-item-content">\n                        <h3 class="'+_List2.default.name+'"><%- summary.name %></h3>\n                        <p class="'+_List2.default.description+"\"><%- summary.description %></p>\n                        <% if (summary.examples != null && summary.examples.length > 0) { %>\n                        <b><%- 'Example' + (summary.examples.length > 1 ? 's' : '') %></b>\n                        <ul>\n                            <% summary.examples.forEach(function(example) { %>\n                            <li>\n                                <a href=\"<%- summary.dashboard + '?ml_toolkit.dataset=' + example.name %>\"><%- example.label %></a>\n                            </li>\n                            <% }) %>\n                        </ul>\n                        <% } %>\n                    </div>\n                </li>\n            <% }) %>\n        </ul>\n    "}),_module3.default.exports=exports.default}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))}).call(exports,__webpack_require__(8)(module))},"contents/List.pcssm":function(module,exports,__webpack_require__){var content=__webpack_require__(830);"string"==typeof content&&(content=[[module.id,content,""]]);__webpack_require__(12)(content,{});content.locals&&(module.exports=content.locals)},830:function(module,exports,__webpack_require__){exports=module.exports=__webpack_require__(11)(),exports.push([module.id,".list------dev---1Map8{list-style:none;padding:0;margin:0}.listItem------dev---1WRGm{padding:5px;margin:5px;display:inline-block;vertical-align:top}.image------dev---1L7eu{height:150px;width:150px;border:1px solid #000}.content------dev---iq7iH{display:inline-block;width:335px;vertical-align:top;margin-left:10px}.name------dev---2D8iz{margin-top:0}.description------dev---12qFX{color:#000}",""]),exports.locals={list:"list------dev---1Map8",listItem:"listItem------dev---1WRGm",image:"image------dev---1L7eu",content:"content------dev---iq7iH",name:"name------dev---2D8iz",description:"description------dev---12qFX"}}});