require(['jquery', 'splunkjs/mvc/simplexml/ready!'], function($) 
{
    	var secondRow = $('#dev1.dashboard-row').first();
	var thirdRow =  $('#dev2.dashboard-row').first();
	var fourthRow =  $('#dev3.dashboard-row').first();
	var fifthRow =  $('#dev4.dashboard-row').first();
	var sixthRow =  $('#dev5.dashboard-row').first();    	

	var panelCells2 = $(secondRow).children('.dashboard-cell');
	var panelCells3 = $(thirdRow).children('.dashboard-cell');
	var panelCells4 = $(fourthRow).children('.dashboard-cell');
	var panelCells5 = $(fifthRow).children('.dashboard-cell');
	var panelCells6 = $(sixthRow).children('.dashboard-cell');
    
	$(panelCells2[0]).css('width', '25%');
	$(panelCells2[1]).css('width', '75%');
	$(panelCells3[0]).css('width', '25%');
        $(panelCells3[1]).css('width', '75%');
	$(panelCells4[0]).css('width', '25%');
        $(panelCells4[1]).css('width', '75%');	
	$(panelCells5[0]).css('width', '25%');
        $(panelCells5[1]).css('width', '75%');
	$(panelCells6[0]).css('width', '25%');
        $(panelCells6[1]).css('width', '75%');
	


});
