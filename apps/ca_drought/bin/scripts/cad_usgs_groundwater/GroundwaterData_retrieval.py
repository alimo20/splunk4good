# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:36:09 2015
Groundwater Retrieval
@author: agoriawala
"""

import requests
from bs4 import BeautifulSoup
import json
import csv


start_date= raw_input('Enter the start date (YYYY-MM-DD) :') #"2015-01-01"
end_date= raw_input('Enter the end date (YYYY-MM-DD) :') #"2015-06-30"


def get_county_codes():
    f=open("./Groundwater/County_codes.csv","rU")
    #reader=csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    f.close()
    n=len(data)
    
    for x in xrange(1,n):
        
        if len(data[x][1])==1:
            data[x][1]="00"+data[x][1]
        if len(data[x][1])==2:
            data[x][1]="0"+data[x][1]
    return data[1:]
    
    
def get_water_level(FIPS_Code,start_date,end_date):
    
    url="http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&countyCd="+FIPS_Code+"&startDT="+start_date+"&endDT="+end_date+"&siteType=GW&wellDepthMin=1&holeDepthMin=1"  
    #print url
    page = requests.get(url)
    soup=BeautifulSoup(page.content)
    #print soup
    text= soup.text.replace("\t\t","\tNULL\t").replace("\t",",").replace(",,",",NULL,").split("\n")
    
    #print text
    if text[0].startswith("Error report"):
        print "Error "+FIPS_Code
        return "Error "
    else : 
        data=[]
        for x in text:
            
            if x.startswith("#") or x.startswith("5s") or len(x)<1:
                continue
            else:
                data.append(x)
    return data
    

def get_location(FIPS_Code,start_date,end_date):
    url="http://waterservices.usgs.gov/nwis/gwlevels/?format=json&countyCd="+FIPS_Code+"&startDT="+start_date+"&endDT="+end_date+"&siteType=GW&wellDepthMin=1&holeDepthMin=1"
    page = requests.get(url)
    o=json.loads(page.content)
    #print o
   
    location={}
    
    n=len(o["value"]["timeSeries"])
    
    for x in xrange(0,n):
        
        site_code= o["value"]["timeSeries"][x]["sourceInfo"]["siteCode"][0]["value"]
        lat=o["value"]["timeSeries"][x]["sourceInfo"]["geoLocation"]["geogLocation"]["latitude"]
        lon=o["value"]["timeSeries"][x]["sourceInfo"]["geoLocation"]["geogLocation"]["longitude"]
        location[site_code]={}
        location[site_code]["lat"]=o["value"]["timeSeries"][x]["sourceInfo"]["geoLocation"]["geogLocation"]["latitude"]
        location[site_code]["long"]=o["value"]["timeSeries"][x]["sourceInfo"]["geoLocation"]["geogLocation"]["longitude"]
    
    return location
    
    
def format_data(county_info,water_info,location):
    county_info[1]=int(county_info[1])
    #print county_info
    #print water_info
    #print location
    index=0
    lev_time=0
    lev_tz=0
    
    formatted_data=[]    
    
    county_headers=["County","FIPS_Code","featureId"]
    location_headers=["Latitude","Longitude"]
    
    n=len(water_info)
    for a in xrange(0,n):
        x=[]
        if a==0:
             temp=water_info[a].split(",")
             
             i=len(temp)
             for h in xrange(0,i):
                 if temp[h]=="lev_va":
                     index=h 
                 if temp[h]=="lev_tm":
                     lev_time=h
                     
                 if temp[h]=="lev_tz_cd":
                    lev_tz=h
             
             x.extend(county_headers)
             x.extend(temp)
             x.extend(location_headers)
             formatted_data.append(x)
        else:
            temp=water_info[a].split(",")
           
            if  temp[index]=="NULL" or temp[index]=="null" or temp[index]=="Null" :
                #print temp
                v=1
            else:
                temp[index]=float(temp[index])
                
            if temp[lev_time]=="NULL" or temp[lev_time]=="null" or temp[lev_time]=="Null" :
                temp[lev_time]="12:00"

            if temp[lev_tz]=="NULL"or temp[lev_tz]=="null" or temp[lev_tz]=="Null" :
                temp[lev_tz]="PST"              
            
            site_no=temp[1]
            #temp[1]=int(temp[1])
            lat=location[site_no]["lat"]
            lon=location[site_no]["long"]
            loc=[lat,lon]
            x.extend(county_info)
            x.extend(temp)
            x.extend(loc)
            formatted_data.append(x)
            
    print
    return formatted_data
    

def save_data(formatted_data,county,start_date,end_date):
    
    #print formatted_data

    year=start_date.split("-")
    year=year[0]
    #print year,county
    op_file=open("./Groundwater/Output_data/"+year+"/"+county+"_"+start_date+"_"+end_date+".csv","wb")
    wr=csv.writer(op_file)
    wr.writerows(formatted_data)    
    
    return
    
def main():
    
    #global start_date,end_date,count
    ERROR_LIST=[]
    county_codes=get_county_codes()
    #print county_codes
    count=0    
    
   # dates=[["2015-01-01","2015-01-31"],["2015-02-01","2015-02-28"],["2015-03-01","2015-03-31"],["2015-04-01","2015-04-30"],["2015-06-01","2015-06-30"],["2015-05-01","2015-05-31"]]
          #["2015-09-01","2014-09-30"],["2014-11-01","2014-11-30"],["2014-07-01","2014-07-31"],["2014-08-01","2014-08-31"],["2014-10-01","2014-10-31"],["2014-12-01","2014-12-31"]]
    
    n=len(county_codes)
    #for date in dates:
     #   start_date=date[0]
      #  end_date=date[1]
    for x in xrange(0,n):
        
        FIPS_Code= "06"+county_codes[x][1]
        #if FIPS_Code=="06071":
         #   continue
        
        print FIPS_Code
        water_info = get_water_level(FIPS_Code,start_date,end_date)
        if water_info=="Error " or len(water_info)<2:
            temp=[FIPS_Code,county_codes[x][0]]
            ERROR_LIST.append(temp)
        else:
            location=get_location(FIPS_Code,start_date,end_date)
            
            formatted_data=format_data(county_codes[x],water_info,location)
            save_data(formatted_data,county_codes[x][0],start_date,end_date)
            count=count+1
    print "Start Date : ",start_date
    print "End Date : ",end_date
    print "Number of counties done :",count
    
    year=start_date.split("-")
    year=year[0]
    resultFile = open("./Groundwater/Output_data/"+year+"Error_list.csv",'wb')
    wr = csv.writer(resultFile)
    wr.writerows(ERROR_LIST)
    print "Number of counties without data :",len(ERROR_LIST)
    print ERROR_LIST

main()