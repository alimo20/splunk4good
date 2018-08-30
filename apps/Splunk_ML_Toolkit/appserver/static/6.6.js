webpackJsonp([6],{"api/layout":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_ARRAY__,__WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_ARRAY__=[__webpack_require__("splunkjs/mvc/layoutview")],__WEBPACK_AMD_DEFINE_RESULT__=function(LayoutView){var _layoutView;return{create:function(options){if(_layoutView)throw new Error("Layout may only be created once");return _layoutView=new LayoutView(options).render(),{getContainerElement:function(){return _layoutView.getContainerElement()}}}}}.apply(exports,__WEBPACK_AMD_DEFINE_ARRAY__),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"splunkjs/mvc/layoutview":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_RESULT__=function(require,exports,module){var $=__webpack_require__("shim/jquery"),_=__webpack_require__("require/underscore"),Backbone=__webpack_require__("require/backbone"),BaseSplunkView=__webpack_require__("splunkjs/mvc/basesplunkview"),HeaderView=__webpack_require__("splunkjs/mvc/headerview"),FooterView=__webpack_require__("splunkjs/mvc/footerview"),template=__webpack_require__(486),splunkUtil=__webpack_require__("shim/splunk.util"),Layout=(__webpack_require__("splunkjs/mvc/sharedmodels"),BaseSplunkView.extend({moduleId:module.id,el:"body",options:{hideChrome:!1,hideAppBar:!1,hideSplunkBar:!1,hideFooter:!1,hideAppsList:!1,layout:"scrolling"},initialize:function(options){this.configure(),BaseSplunkView.prototype.initialize.apply(this,arguments),this.$el.removeAttr("class").removeAttr("id")},getContainerElement:function(){if(this._$main)return this._$main[0];throw new Error("Layout must be rendered before container can be accessed")},render:function(){var compiledTemplate=_.template(this.template);this.$el.append(compiledTemplate({_:_,make_url:splunkUtil.make_url})),this._$main=$('<div role="main">'),this.$("#navSkip").after(this._$main),this._applyLayoutStyles();var $header=this.$("header");if(!this.options.hideChrome&&(this._headerView=new HeaderView({id:"header",el:$header,splunkbar:!this.options.hideSplunkBar,appbar:!this.options.hideAppBar,showAppsList:!this.options.hideAppsList}).render(),$header.removeAttr("class").removeAttr("id"),!this.options.hideFooter)){var $footer=this.$("footer");this._footerView=new FooterView({id:"footer",el:$footer}).render(),$footer.removeAttr("class").removeAttr("id")}return this},remove:function(){return this._$main=null,this._headerView.remove(),this._footerView.remove(),BaseSplunkView.prototype.remove.apply(this,arguments)},setElement:function(){return Backbone.View.prototype.setElement.apply(this,arguments)},_applyLayoutStyles:function(){this.$el.css({margin:0}),"fixed"===this.options.layout?(this._$main.css({flex:"1 0 0",position:"relative"}),this.$el.css({display:"flex",flexDirection:"column",position:"fixed",left:0,top:0,right:0,bottom:0,overflow:"hidden"}),this.$el.find("header, footer").css({flex:"0 0 auto"})):this._$main.css({position:"relative",minHeight:"500px"})},template:template}));return Layout}.call(exports,__webpack_require__,exports,module),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"splunkjs/mvc/headerview":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_RESULT__=function(require,exports,module){var $=__webpack_require__("shim/jquery"),BaseSplunkView=(__webpack_require__("splunkjs/mvc/mvc"),__webpack_require__("splunkjs/mvc/basesplunkview")),GlobalNav=__webpack_require__("views/shared/splunkbar/Master"),AppNav=__webpack_require__("views/shared/appbar/Master"),SideNav=__webpack_require__("views/shared/litebar/Master"),sharedModels=__webpack_require__("splunkjs/mvc/sharedmodels"),ConfigModel=(__webpack_require__("shim/splunk.util"),__webpack_require__("models/config")),HeaderView=(__webpack_require__("uri/route"),BaseSplunkView.extend({moduleId:module.id,className:"splunk-header",options:{appbar:!0,splunkbar:!0,useSessionStorageCache:!1,acceleratedAppNav:!1},initialize:function(){this.configure(),this.model=this.model||{},this.model.application=sharedModels.get("app"),this.model.user=sharedModels.get("user"),this.model.appLocal=sharedModels.get("appLocal"),this.model.serverInfo=sharedModels.get("serverInfo"),this.model.userPref=sharedModels.get("userPref"),this.collection=this.collection||{},this.collection.appLocals=sharedModels.get("appLocals"),this.collection.tours=sharedModels.get("tours"),this.deferreds=this.deferreds||{},this.deferreds.tours=$.Deferred(),$.when(this.model.serverInfo.dfd).then(function(){this.useSideNav=this.settings.get("litebar")||this.model.serverInfo.isLite(),this.useSideNav&&this.collection.tours.dfd.done(function(){this.sideNav=SideNav.create({model:{application:this.model.application,appLocal:this.model.appLocal,user:this.model.user,serverInfo:this.model.serverInfo,config:ConfigModel,appNav:this.model.appNav,userPref:this.model.userPref},collection:{apps:this.collection.appLocals,tours:this.collection.tours}}),this.deferreds.tours.resolve()}.bind(this))}.bind(this))},renderSplunkBar:function(){this.settings.get("splunkbar")?(this.globalNav||(this.globalNav=GlobalNav.create({showAppsList:this.settings.get("showAppsList")!==!1,model:{application:this.model.application,appLocal:this.model.appLocal,user:this.model.user,serverInfo:this.model.serverInfo,config:ConfigModel,userPref:this.model.userPref},collection:{apps:this.collection.appLocals}})),this.globalNav.render().prependTo(this.$el)):this.globalNav&&(this.globalNav.remove(),this.globalNav=null)},renderAppBar:function(){this.settings.get("appbar")?(this.appNav||(this.appNav=AppNav.create({model:{application:this.model.application,app:this.model.appLocal,user:this.model.user,serverInfo:this.model.serverInfo,appNav:this.model.appNav},useSessionStorageCache:this.settings.get("useSessionStorageCache"),autoRender:!1})),this.appNav.render().appendTo(this.$el)):this.appNav&&(this.appNav.remove(),this.appNav=null)},render:function(){return this.stopListening(this.settings,"change:splunkbar"),this.stopListening(this.settings,"change:appbar"),$.when(this.model.serverInfo.dfd).then(function(){this.useSideNav?$.when(this.deferreds.tours).then(function(){this.$el.empty(),this.$el.append(this.sideNav.el)}.bind(this)):(this.$el.empty(),this.renderSplunkBar(),this.renderAppBar(),this.listenTo(this.settings,"change:appbar",function(){this.renderAppBar()}),this.listenTo(this.settings,"change:splunkbar",function(){this.renderSplunkBar()}))}.bind(this)),this}}));return HeaderView}.call(exports,__webpack_require__,exports,module),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},"splunkjs/mvc/footerview":function(module,exports,__webpack_require__){var __WEBPACK_AMD_DEFINE_RESULT__;__WEBPACK_AMD_DEFINE_RESULT__=function(require,exports,module){var $=__webpack_require__("shim/jquery"),_=__webpack_require__("require/underscore"),BaseSplunkView=(__webpack_require__("splunkjs/mvc/mvc"),__webpack_require__("splunkjs/mvc/basesplunkview")),Footer=__webpack_require__("views/shared/footer/Master"),sharedModels=__webpack_require__("splunkjs/mvc/sharedmodels"),FooterView=BaseSplunkView.extend({moduleId:module.id,className:"splunk-footer",initialize:function(){var appModel=sharedModels.get("app"),appLocalModel=sharedModels.get("appLocal"),serverInfoModel=sharedModels.get("serverInfo"),appLocals=sharedModels.get("appLocals");this.dfd=$.when.apply($,[appModel.dfd,appLocalModel.dfd,serverInfoModel.dfd,appLocals.dfd]),this.dfd.done(_.bind(function(){this.footer=Footer.create({model:{application:appModel,appLocal:appLocalModel,serverInfo:serverInfoModel},collection:{apps:appLocals}})},this))},render:function(){return this.dfd.done(_.bind(function(){this.$el.append(this.footer.render().el)},this)),this}});return FooterView}.call(exports,__webpack_require__,exports,module),!(void 0!==__WEBPACK_AMD_DEFINE_RESULT__&&(module.exports=__WEBPACK_AMD_DEFINE_RESULT__))},486:function(module,exports){module.exports="<style>\n    /*  Fonts */\n    /*  ----------- */\n\n    /*  Regular */\n    @font-face {\n        font-family: 'Roboto';\n        src: url('<%- make_url(\"static/fonts/roboto-regular-webfont.woff\") %>') format('woff');\n        font-weight: normal;\n        font-style: normal;\n    }\n\n    /*  Bold */\n    @font-face {\n        font-family: 'Roboto';\n        src: url('<%- make_url(\"static/fonts/roboto-bold-webfont.woff\") %>') format('woff');\n        font-weight: bold;\n        font-style: normal;\n    }\n\n    /*  light */\n    @font-face {\n        font-family: 'Roboto';\n        src: url('<%- make_url(\"static/fonts/roboto-light-webfont.woff\") %>') format('woff');\n        font-weight: 200;\n        font-style: normal;\n    }\n\n    @font-face {\n      font-family: 'Architects Daughter';\n      font-style: normal;\n      font-weight: normal;\n      src: url('<%- make_url(\"static/fonts/ArchitectsDaughter.ttf\") %>') format('truetype');\n    }\n</style>\n<a style=\"position:absolute; top:-100px; left:-1000px\" href=\"#navSkip\" tabIndex=\"1\">\n    <%- _(\"Screen reader users, click here to skip the navigation bar\").t() %>\n</a>\n<header></header>\n<a id=\"navSkip\"></a>\n<footer></footer>\n"}});