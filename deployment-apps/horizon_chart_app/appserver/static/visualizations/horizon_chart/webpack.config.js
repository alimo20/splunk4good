var webpack = require('webpack');
var path = require('path');

module.exports = {
    entry: 'horizon_chart',
    resolve: {
        root: [
            path.join(__dirname, 'src'),
        ]
    },
    output: {
        filename: 'visualization.js',
        libraryTarget: 'amd'
    },
    module: {
        loaders: [
            {
                test: /cubism\.v1\.js$/,
                loader: 'imports-loader?window=>{addEventListener:function(){}}&d3=d3'
            }
        ]
    },
    externals: [
        'api/SplunkVisualizationBase',
        'api/SplunkVisualizationUtils'
    ]
};