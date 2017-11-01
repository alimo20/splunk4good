define(["jquery","underscore","moment"],function(e,s,t){return function(e){function s(i){if(t[i])return t[i].exports;var n=t[i]={exports:{},id:i,loaded:!1};return e[i].call(n.exports,n,n.exports,s),n.loaded=!0,n.exports}var t={};return s.m=e,s.c=t,s.p="",s(0)}({0:function(e,s,t){function i(e,s){if(!(e instanceof s))throw new TypeError("Cannot call a class as a function")}var n,r,o=function(){function e(e,s){for(var t=0;t<s.length;t++){var i=s[t];i.enumerable=i.enumerable||!1,i.configurable=!0,"value"in i&&(i.writable=!0),Object.defineProperty(e,i.key,i)}}return function(s,t,i){return t&&e(s.prototype,t),i&&e(s,i),s}}();n=[t(1),t(3),t(4),t(5),t(38),t(39),t(40)],r=function(e,s,t,n,r,a){"use strict";var c=a("2038-01-19T03:14:07Z"),l={CloudTrail:"aws:cloudtrail",Config:"aws:config",S3AccessLogs:"aws:s3:accesslogs",CloudFrontAccessLogs:"aws:cloudfront:accesslogs",ELBAccessLogs:"aws:elb:accesslogs",CustomLogs:"aws:s3"},u=function(){function u(e,s,n,r){i(this,u),this.context=t.getComponent(e,s),this.model=n,this.serviceName=s,this.util=r,this._conflict_fields=JSON.parse(this.model.get("_conflict_fields")||"{}")}return o(u,[{key:"onCreate",value:function(){var e=this._getInputService(),t=e.input,i=e.service,r=n.map.find(function(e){return e.service===i&&e.input===t});if(r){var o=s.omit(r,["service","input"]);this.model.set(o)}if(Object.keys(this._conflict_fields).length>0&&this.model.set(this._conflict_fields,{unset:!0}),"aws_billing"===this.serviceName){if(s.isUndefined(this.model.get("initial_scan_datetime"))){var c=a().utc().subtract(3,"months").startOf("month");this.model.set("initial_scan_datetime",c.format("YYYY-MM-DDTHH:mm:ss")+"Z")}}else"aws_s3"===this.serviceName?s.isUndefined(this.model.get("initial_scan_datetime"))&&this.model.set("initial_scan_datetime",a().utc().format("YYYY-MM-DDTHH:mm:ss")+"Z"):"aws_sqs_based_s3"===this.serviceName&&(this.model.on("change:s3_file_decoder",function(){this.model.set("sourcetype",l[this.model.get("s3_file_decoder")])},this),s.isUndefined(this.model.get("sourcetype"))&&this.model.set("sourcetype",l[this.model.get("s3_file_decoder")]))}},{key:"onRender",value:function(){var s=this,t=this._getInputService(),i=t.input,n=t.service,o=e(".section-header > .section-label");if(o.length>0){var a=r.buildInputLink(n);e(o[0]).append(a)}if(Object.keys(this._conflict_fields).length>0){var c=this.context.entity,l="The following fields have conflict. ";l+=Object.keys(this._conflict_fields).map(function(e){s.util.addErrorToComponent(s.serviceName,e);var t=c.find(function(s){return s.field===e}).label;return t+" = ["+s._conflict_fields[e].join(", ")+"]"}).join(", "),l+=". Please resolve the conflict in conf file first.",this.util.displayErrorMsg(l)}"aws_s3"===this.serviceName?("aws_cloudtrail"!==i&&this._hideFields("ct_blacklist"),"others"!==i&&(this._hideFields(["whitelist","blacklist"]),this._disableFields("sourcetype"))):"splunk_ta_aws_logs"===this.serviceName?this._disableFields(["log_type","sourcetype"]):"aws_sqs_based_s3"===this.serviceName?"others"!==i&&this._disableFields(["s3_file_decoder","sourcetype"]):"aws_kinesis"===this.serviceName&&"vpc_flow_logs"===i&&this._disableFields(["format","encoding","sourcetype"])}},{key:"onSave",value:function(){if("splunk_ta_aws_logs"===this.serviceName){if("cloudfront:accesslogs"===this.model.get("log_type")&&!this.model.has("log_name_format"))return this.util.displayErrorMsg('Field "Distribution ID" is required.'),!1}else if("aws_cloudwatch"===this.serviceName){if(this.model.has("period")&&this.model.has("polling_interval")){var e=this.model.get("polling_interval")/this.model.get("period");if(Math.floor(e)!==e)return this.util.displayErrorMsg('The number of "Polling Interval" field should be a multiple of "Granularity".'),!1}}else if("aws_s3"===this.serviceName&&this.model.has("terminal_scan_datetime")){var s=this.model.get("terminal_scan_datetime");try{if(s=a(s),s>c)return this.util.displayErrorMsg("The max supported timestamp is 2038-01-19T03:14:07Z due to Year 2038 problem"),this.util.addErrorToComponent(this.serviceName,"terminal_scan_datetime"),!1;this.util.removeErrorMsg(),this.util.removeErrorFromComponent(this.serviceName,"terminal_scan_datetime")}catch(e){return!1}}return!0}},{key:"_getInputService",value:function(){this.urlParams=this._extractQuery();var e=this.urlParams,s=e.input,t=e.service;return s||(s=n.detectSource(this.model,this.serviceName)),{input:s,service:t}}},{key:"_extractQuery",value:function(){var e=document.location.search;if(e&&0!==e.length){var s=document.location.search.substring(1).split("&"),t={};return s.forEach(function(e){var s=e.split("=");1===s.length?t[s[0]]=null:t[s[0]]=s[1]}),t}}},{key:"_disableFields",value:function(t){s.isArray(t)||(t=[t]);var i=!0,n=!1,r=void 0;try{for(var o,a=t[Symbol.iterator]();!(i=(o=a.next()).done);i=!0){var c=o.value;e('[data-name="'+c+'"]').find("input").prop("disabled",!0)}}catch(e){n=!0,r=e}finally{try{!i&&a.return&&a.return()}finally{if(n)throw r}}}},{key:"_hideFields",value:function(t){s.isArray(t)||(t=[t]);var i=!0,n=!1,r=void 0;try{for(var o,a=t[Symbol.iterator]();!(i=(o=a.next()).done);i=!0){var c=o.value;e('[data-name="'+c+'"]').parents(".form-horizontal."+c).hide()}}catch(e){n=!0,r=e}finally{try{!i&&a.return&&a.return()}finally{if(n)throw r}}}}]),u}();return u}.apply(s,n),!(void 0!==r&&(e.exports=r))},1:function(s,t){s.exports=e},3:function(e,t){e.exports=s},4:function(e,s,t){var i,n;i=[],n=function(){"use strict";return{getField:function(e,s,t){var i=this.getComponent(e,s).entity;return i.find(function(e){return e.field===t})},getComponent:function(e,s){var t=this.getServices(e),i=t.find(function(e){return e.name===s});return i},getServices:function(e){return e.pages.inputs.services}}}.apply(s,i),!(void 0!==n&&(e.exports=n))},5:function(e,s,t){var i,n;i=[t(3)],n=function(e){"use strict";return{mockService:{vpc_flow_logs:"VPC Flow Logs",s3_access_logs:"S3 Access Logs",cloudfront_access_logs:"Cloudfront Access Logs",elb_access_logs:"ELB Access Logs",others:"Custom Data Type"},map:[{service:"aws_kinesis",input:"vpc_flow_logs",format:"CloudWatchLogs",sourcetype:"aws:cloudwatchlogs:vpcflow"},{service:"aws_cloudwatch_logs",input:"vpc_flow_logs",sourcetype:"aws:cloudwatchlogs:vpcflow"},{service:"aws_s3",input:"aws_cloudtrail",sourcetype:"aws:cloudtrail"},{service:"aws_s3",input:"cloudfront_access_logs",sourcetype:"aws:cloudfront:accesslogs"},{service:"aws_s3",input:"elb_access_logs",sourcetype:"aws:elb:accesslogs"},{service:"aws_s3",input:"s3_access_logs",sourcetype:"aws:s3:accesslogs"},{service:"splunk_ta_aws_logs",input:"aws_cloudtrail",log_type:"cloudtrail"},{service:"splunk_ta_aws_logs",input:"cloudfront_access_logs",log_type:"cloudfront:accesslogs"},{service:"splunk_ta_aws_logs",input:"elb_access_logs",log_type:"elb:accesslogs"},{service:"splunk_ta_aws_logs",input:"s3_access_logs",log_type:"s3:accesslogs"},{service:"aws_sqs_based_s3",input:"aws_config",s3_file_decoder:"Config"},{service:"aws_sqs_based_s3",input:"aws_cloudtrail",s3_file_decoder:"CloudTrail"},{service:"aws_sqs_based_s3",input:"cloudfront_access_logs",s3_file_decoder:"CloudFrontAccessLogs"},{service:"aws_sqs_based_s3",input:"elb_access_logs",s3_file_decoder:"ELBAccessLogs"},{service:"aws_sqs_based_s3",input:"s3_access_logs",s3_file_decoder:"S3AccessLogs"},{service:"aws_sqs_based_s3",input:"others",s3_file_decoder:"CustomLogs"}],detectSource:function(s,t){var i=this.map.filter(function(e){return e.service===t}),n=i.find(function(t){t=e.omit(t,["service","input"]);var i=e.pick(s.attributes,Object.keys(t));return e.isEqual(t,i)});return n&&"input"in n?n.input:t}}}.apply(s,i),!(void 0!==n&&(e.exports=n))},8:function(e,s){e.exports=function(){var e=[];return e.toString=function(){for(var e=[],s=0;s<this.length;s++){var t=this[s];t[2]?e.push("@media "+t[2]+"{"+t[1]+"}"):e.push(t[1])}return e.join("")},e.i=function(s,t){"string"==typeof s&&(s=[[null,s,""]]);for(var i={},n=0;n<this.length;n++){var r=this[n][0];"number"==typeof r&&(i[r]=!0)}for(n=0;n<s.length;n++){var o=s[n];"number"==typeof o[0]&&i[o[0]]||(t&&!o[2]?o[2]=t:t&&(o[2]="("+o[2]+") and ("+t+")"),e.push(o))}},e}},9:function(e,s,t){function i(e,s){for(var t=0;t<e.length;t++){var i=e[t],n=p[i.id];if(n){n.refs++;for(var r=0;r<n.parts.length;r++)n.parts[r](i.parts[r]);for(;r<i.parts.length;r++)n.parts.push(l(i.parts[r],s))}else{for(var o=[],r=0;r<i.parts.length;r++)o.push(l(i.parts[r],s));p[i.id]={id:i.id,refs:1,parts:o}}}}function n(e){for(var s=[],t={},i=0;i<e.length;i++){var n=e[i],r=n[0],o=n[1],a=n[2],c=n[3],l={css:o,media:a,sourceMap:c};t[r]?t[r].parts.push(l):s.push(t[r]={id:r,parts:[l]})}return s}function r(e,s){var t=v(),i=w[w.length-1];if("top"===e.insertAt)i?i.nextSibling?t.insertBefore(s,i.nextSibling):t.appendChild(s):t.insertBefore(s,t.firstChild),w.push(s);else{if("bottom"!==e.insertAt)throw new Error("Invalid value for parameter 'insertAt'. Must be 'top' or 'bottom'.");t.appendChild(s)}}function o(e){e.parentNode.removeChild(e);var s=w.indexOf(e);s>=0&&w.splice(s,1)}function a(e){var s=document.createElement("style");return s.type="text/css",r(e,s),s}function c(e){var s=document.createElement("link");return s.rel="stylesheet",r(e,s),s}function l(e,s){var t,i,n;if(s.singleton){var r=m++;t=g||(g=a(s)),i=u.bind(null,t,r,!1),n=u.bind(null,t,r,!0)}else e.sourceMap&&"function"==typeof URL&&"function"==typeof URL.createObjectURL&&"function"==typeof URL.revokeObjectURL&&"function"==typeof Blob&&"function"==typeof btoa?(t=c(s),i=f.bind(null,t),n=function(){o(t),t.href&&URL.revokeObjectURL(t.href)}):(t=a(s),i=d.bind(null,t),n=function(){o(t)});return i(e),function(s){if(s){if(s.css===e.css&&s.media===e.media&&s.sourceMap===e.sourceMap)return;i(e=s)}else n()}}function u(e,s,t,i){var n=t?"":i.css;if(e.styleSheet)e.styleSheet.cssText=y(s,n);else{var r=document.createTextNode(n),o=e.childNodes;o[s]&&e.removeChild(o[s]),o.length?e.insertBefore(r,o[s]):e.appendChild(r)}}function d(e,s){var t=s.css,i=s.media;if(i&&e.setAttribute("media",i),e.styleSheet)e.styleSheet.cssText=t;else{for(;e.firstChild;)e.removeChild(e.firstChild);e.appendChild(document.createTextNode(t))}}function f(e,s){var t=s.css,i=s.sourceMap;i&&(t+="\n/*# sourceMappingURL=data:application/json;base64,"+btoa(unescape(encodeURIComponent(JSON.stringify(i))))+" */");var n=new Blob([t],{type:"text/css"}),r=e.href;e.href=URL.createObjectURL(n),r&&URL.revokeObjectURL(r)}var p={},_=function(e){var s;return function(){return"undefined"==typeof s&&(s=e.apply(this,arguments)),s}},h=_(function(){return/msie [6-9]\b/.test(window.navigator.userAgent.toLowerCase())}),v=_(function(){return document.head||document.getElementsByTagName("head")[0]}),g=null,m=0,w=[];e.exports=function(e,s){s=s||{},"undefined"==typeof s.singleton&&(s.singleton=h()),"undefined"==typeof s.insertAt&&(s.insertAt="bottom");var t=n(e);return i(t,s),function(e){for(var r=[],o=0;o<t.length;o++){var a=t[o],c=p[a.id];c.refs--,r.push(c)}if(e){var l=n(e);i(l,s)}for(var o=0;o<r.length;o++){var c=r[o];if(0===c.refs){for(var u=0;u<c.parts.length;u++)c.parts[u]();delete p[c.id]}}}};var y=function(){var e=[];return function(s,t){return e[s]=t,e.filter(Boolean).join("\n")}}()},38:function(e,s,t){var i,n;i=[t(3)],n=function(e){"use strict";var s={aws_description:"aws.description",aws_config:"aws.config",aws_config_rule:"aws.configrules",aws_cloudwatch:"aws.cloudwatch",aws_cloudwatch_logs:"aws.cloudwatchlogs",aws_s3:"aws.s3",splunk_ta_aws_logs:"aws.incrementals3",aws_sqs_based_s3:"aws.sqs_based_s3",aws_billing:"aws.billing",aws_cloudtrail:"aws.cloudtrail",aws_kinesis:"aws.kinesis",aws_inspector:"aws.inspector",splunk_ta_aws_sqs:"aws.sqs"};return{buildLinkNode:function(s,t){return t=t||e("Learn more").t(),'<a class="external" target="_blank" href="/help?location='+encodeURIComponent("[AddOns:released]")+s+'">'+t+"</a>"},buildInputLink:function(e){return this.buildLinkNode(s[e])},ALL_LINKS:{SQS_BASED_S3:"aws.configure_aws.configure_sqs"}}}.apply(s,i),!(void 0!==n&&(e.exports=n))},39:function(e,s){e.exports=t},40:function(e,s,t){var i=t(41);"string"==typeof i&&(i=[[e.id,i,""]]);t(9)(i,{});i.locals&&(e.exports=i.locals)},41:function(e,s,t){s=e.exports=t(8)(),s.push([e.id,".section-label .external{font-size:10px}",""])}})});