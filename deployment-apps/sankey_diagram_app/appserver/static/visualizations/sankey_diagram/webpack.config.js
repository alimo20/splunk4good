var webpack = require('webpack');
var path = require('path');

module.exports = {
    entry: 'sankey_diagram',
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
                test: /sankey\.js$/,
                loader: 'imports-loader?d3=d3'
            }
        ]
    },
    externals: [
        'api/SplunkVisualizationBase',
        'api/SplunkVisualizationUtils'
    ]
};