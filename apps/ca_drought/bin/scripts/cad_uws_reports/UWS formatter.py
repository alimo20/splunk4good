# -*- coding: utf-8 -*-
"""
This file formats Urban Water supplier reports
into a format suitable to be indexed by Splunk.
This script will need updates for 2016 data.

IMPORTANT: this script takes CSV files as input.
You will need to change the files downloaded from source
from XLS to CSV. 
"""

import csv
import sys
import datetime

     
def get_csv_data():
    
    csv_file= raw_input('Enter csv file name alongwith the extension : ') #Eg "uw_supplier_data070115.csv"
    
    f=open(csv_file,"rU")
    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    return data
    

def get_header_dict(col_list):
    n=len(col_list)
    col_names={}    
    for i in xrange(0,n):
        col_names[col_list[i].strip()]=i
    
    return col_names

def get_current_data(month,index,excel_data):
    
    current_data=[]
    current_data.append(excel_data[0])
    for row in excel_data:
        if row[index]==month:
            current_data.append(row)
    print len(current_data)
    return current_data

def format_current_data(current_data,header_dict,supp_unit):
        
        formatted_data=[]
        final=[]
        
        #Format column headers
        #print current_data[0]
        #print 
        
        
        current_data[0][header_dict["CALCULATED R-GPCD 2014/2015 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]]="CALCULATED R-GPCD 2014/2015"
        current_data[0][header_dict["REPORTED Residential Gallons-per-Capita-Day (R-GPCD) (starting in September 2014)"]]="REPORTED Residential Gallons-per-Capita-Day"
        
        if "CALCULATED R-GPCD 2013 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)" in header_dict.keys():
            current_data[0][header_dict["CALCULATED R-GPCD 2013 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]]="CALCULATED R-GPCD 2013"
        else:
            current_data[0].append("CALCULATED R-GPCD 2013")
        
        k=len(current_data[0])    
        for j in xrange(0,k):
            
            current_data[0][j]=current_data[0][j].replace(" CALCULATED Production Monthly Gallons Month ","CALCULATED Monthly Production Gallons ")
            current_data[0][j]=current_data[0][j].strip().replace(" ","_")
            current_data[0][j]=current_data[0][j].replace("/","_")
            current_data[0][j]=current_data[0][j].replace("%","Percent")
            current_data[0][j]=current_data[0][j].replace("_-_","_")
            current_data[0][j]=current_data[0][j].replace("-","_")
            
        #print current_data[0]    
        formatted_data.append(current_data[0])
        

        #Format values
        for row in current_data[1:]:
            #print row                
            
            #Changes Reporting month format to YY-M from M-YY to match Splunk indexed data
            """month=row[header_dict["Reporting Month"]]
            month=month.split("-")
            #print month
            temp=month[0]+"-"+month[1]
            #print temp
            row[header_dict["Reporting Month"]]=temp"""
            
            #Converts all integers being represented as String back to integers
            row[header_dict["Total Monthly Potable Water Production 2014/2015"]]=  float(row[header_dict["Total Monthly Potable Water Production 2014/2015"]].replace(",",""))
            row[header_dict["Total Monthly Potable Water Production 2013"]]=  float(row[header_dict["Total Monthly Potable Water Production 2013"]].replace(",",""))
            row[header_dict["Total Population Served"]]=  long(row[header_dict["Total Population Served"]].replace(",",""))
            row[header_dict["CALCULATED Production Monthly Gallons Month 2013"]]=long(row[header_dict["CALCULATED Production Monthly Gallons Month 2013"]].replace(",",""))
            row[header_dict["CALCULATED Production Monthly Gallons Month 2014/2015"]]=long(row[header_dict["CALCULATED Production Monthly Gallons Month 2014/2015"]].replace(",",""))        
            row[header_dict["CALCULATED R-GPCD 2014/2015 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]]=float(row[header_dict["CALCULATED R-GPCD 2014/2015 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]])         
            
            #Formatting the field to be between 0-1
            temp_value=row[header_dict["% Residential Use"]]
            temp_value=temp_value.replace("%","")
            temp_value=float(temp_value)
            
            #if temp_value>=0.0 and temp_value<=1.0:
               # row[header_dict["% Residential Use"]]=temp_value
            #else:
            row[header_dict["% Residential Use"]]=float(temp_value/100)
            
            if "CALCULATED R-GPCD 2013 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)" in header_dict.keys() :  #May not be present
                row[header_dict["CALCULATED R-GPCD 2013 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]]=float(row[header_dict["CALCULATED R-GPCD 2013 (Values calculated by Water Board staff using methodology available at http://www.waterboards.ca.gov/waterrights/water_issues/programs/drought/docs/ws_tools/guidance_estimate_res_gpcd.pdf)"]].replace(",",""))        
            else:
                TMP_13=row[header_dict["Total Monthly Potable Water Production 2013"]]
                RGPCD_2013=calculate_RGPCD(TMP_13,row,header_dict,supp_unit)
                row.append(RGPCD_2013)
               
            
            if len(row[header_dict["REPORTED Residential Gallons-per-Capita-Day (R-GPCD) (starting in September 2014)"]])>0:
                row[header_dict["REPORTED Residential Gallons-per-Capita-Day (R-GPCD) (starting in September 2014)"]]=  float(row[header_dict["REPORTED Residential Gallons-per-Capita-Day (R-GPCD) (starting in September 2014)"]])
            else:
                #Calculate R-GCPD
                TMP_14_15=row[header_dict["Total Monthly Potable Water Production 2014/2015"]]
                RGPCD_2014_2015=calculate_RGPCD(TMP_14_15,row,header_dict,supp_unit)
                row[header_dict["REPORTED Residential Gallons-per-Capita-Day (R-GPCD) (starting in September 2014)"]]=RGPCD_2014_2015
                
            #Recycled water & Units field formatting            
            if len(row[header_dict["Optional - Recycled Water"]])<1:
               row[header_dict["Optional - Recycled Water"]]=-99999
            
            if len(row[header_dict["Recycled Water Units"]])<1:
                row[header_dict["Recycled Water Units"]]="NULL"
            
            formatted_data.append(row)
        
        #print formatted_data[0:5]
        #Moving Reporting_Month ahead
        
        index=header_dict["Reporting Month"]        
        for row in formatted_data:
            
            value=row[index]
            row.insert(0,value)
            row.pop(index+1)
            
            if value=="Reporting_Month":
                row.insert(1,"Timestamp")
            else:
                date=datetime.datetime.strptime(value, "%y-%b").date()
                row.insert(1,date.strftime("%m/%d/%Y %H:%M" ))
                
            final.append(row)

        return final 
        

def calculate_RGPCD(TMP,row,header_dict,supp_unit):
    
    unit_conv_factors={"G":1,"MG":1000000,"CCF":748,"AF":325851}
    
    PRU= row[header_dict["% Residential Use"]]
    Units = row[header_dict["Units"]]
    
    if Units=='corrected':
        Supplier_name=row[header_dict["Supplier Name"]]
        Units=supp_unit[Supplier_name]
    
    C = unit_conv_factors[Units]
    TPS = row[header_dict["Total Population Served"]]
    month = row[header_dict["Reporting Month"]]
    no_of_days = get_no_of_days(month)
    
    R_GPCD= ((TMP*PRU*C)/TPS)/no_of_days
    
    return round(R_GPCD,1)
    
    
def get_no_of_days(month):
    month=month.split("-")
    Y=month[0]
    M=month[1]
    
    if M=="Feb":
        if Y=="16" or Y=="20":
            return 29.0
        else:
            return 28.0
    
    if M=="Jan" or M=="Mar" or M=="May" or M=="Jul" or M=="Aug" or M=="Oct" or M=="Dec":
        return 31.0
    
    if M=="Apr" or M=="Jun" or M=="Sep" or M=="Nov":    
        return 30.0


def write_data(formatted_data,reporting_month) :
    
    op_file=open("./Formatted Output/UWS_report "+reporting_month+".csv","wb")
    wr=csv.writer(op_file)
    wr.writerows(formatted_data)    
    
    return


def get_units():
    
    supp_unit={}
    csv_file="Supplier_Units_lookup.csv"
    
    f=open(csv_file,"rU")
    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    
    for row in data:
        supp_unit[row[0]]=row[1]
    
    #print supp_unit
    return supp_unit  

                
def main():
    print
    
    #for r in ["14-Jun","14-Jul","14-Aug","14-Sep","14-Oct","14-Nov","14-Dec","15-Jan","15-Feb","15-Mar","15-Apr","15-May"]:
        #reporting_month=r
    reporting_month = raw_input('Enter the reporting month (format must match the string under Reporting Month col. Eg 15-Mar) : ')
        
    excel_data=get_csv_data()
    #print excel_data[0]
    header_dict=get_header_dict(excel_data[0])
    #print header_dict
    current_data=get_current_data(reporting_month,header_dict["Reporting Month"],excel_data)
    #print current_data
    
    supp_unit=get_units()    
    
    formatted_data=format_current_data(current_data,header_dict,supp_unit)
    #print formatted_data[0:5]
    write_data(formatted_data,reporting_month)

    
main()