require([
    'underscore',
    'jquery',
    'splunkjs/mvc',
    'splunkjs/mvc/tableview',
    'splunkjs/mvc/simplexml/ready!'
], function(_, $, mvc, TableView) {

     // Row Coloring Example with custom, client-side range interpretation
    var CustomRangeRenderer = TableView.BaseCellRenderer.extend({
        canRender: function(cell) {
            // Enable this custom cell renderer for both the active_hist_searches and the active_realtime_searches field
            return _(['result_code', ]).contains(cell.field);
        },

        render: function($td, cell) {
            // Add a class to the cell based on the returned value
            var value = parseFloat(cell.value);
            var updated = 0;

            // Apply interpretation for number of historical searches
            if (cell.field === 'result_code') {
                if (value == 0) {
                    $td.addClass('range-cell').addClass('range-ok');
                    updated = 1;
                }
                else if (value == 103) {
                    $td.addClass('range-cell').addClass('range-error');
                    updated = 1;
                }
            }

            // Update the cell content
            if (updated==1)
              $td.text(value.toFixed(0)).addClass('numeric');
            else
              $td.text(cell.value);

        }
    });
    mvc.Components.get('forensic').getVisualization(function(tableView) {
        // Add custom cell renderer, the table will re-render automatically.
        tableView.addCellRenderer(new CustomRangeRenderer());
    });
});