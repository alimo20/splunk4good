/* TODO: jink to replace theme_utils with that from core */
require.config({
  paths: {
    theme_utils: '../app/simple_xml_examples/theme_utils'
  }
});

require([
    'underscore',
    'jquery',
    'splunkjs/mvc',
    'splunkjs/mvc/tableview',
    'theme_utils',
    'splunkjs/mvc/simplexml/ready!'
], function(_, $, mvc, TableView, themeUtils) {

    var isDarkTheme = themeUtils.getCurrentTheme && themeUtils.getCurrentTheme() === 'dark';

     // Row Coloring Example with custom, client-side range interpretation

    var CustomRangeRenderer = TableView.BaseCellRenderer.extend({
        canRender: function(cell) {
            // Enable this custom cell renderer for both the active_hist_searches and the active_realtime_searches field
            return _(['active_hist_searches', 'active_realtime_searches']).contains(cell.field);
        },
        render: function($td, cell) {
            // Add a class to the cell based on the returned value
            var value = parseFloat(cell.value);

            // Apply interpretation for number of historical searches
            if (cell.field === 'active_hist_searches') {
                if (value > 2) {
                    $td.addClass('range-cell').addClass('range-elevated');
                }
            }

            // Apply interpretation for number of realtime searches
            if (cell.field === 'active_realtime_searches') {
                if (value > 1) {
                    $td.addClass('range-cell').addClass('range-severe');
                }
            }

            // Update the cell content
            $td.text(value.toFixed(2)).addClass('numeric');
        }
    });

    mvc.Components.get('highlight').getVisualization(function(tableView) {
        tableView.on('rendered', function() {
            // Apply class of the cells to the parent row in order to color the whole row
            tableView.$el.find('td.range-cell').each(function() {
                $(this).parents('tr').addClass(this.className);
            });

            if(isDarkTheme){
              tableView.$el.find('td.timestamp').each(function() {
                 $(this).parents('tr').addClass('dark');
              });
            }
        });
        // Add custom cell renderer, the table will re-render automatically.
        tableView.addCellRenderer(new CustomRangeRenderer());
    });

});