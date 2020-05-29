##List of Functions (Update as more added in)
#startTimes
#failTimes
#removeDuplicates
#getClosestandStartTimes
#timeOnOff
#shutTimes



import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf,PandasUDFType

schema2 = StructType([
    StructField("CSN", StringType()),
    StructField("EventID", LongType()),
    StructField("DateTime", StringType())
])

 

@pandas_udf(schema2, functionType=PandasUDFType.GROUPED_MAP)
def startTimes(df):
    gr = df['CSN'].iloc[0]#used for the default returned dataframe if no data being returned
    EventID = 65533
    events = df[df['EventID']==EventID].sort_values(['DateTime']) #start events
    DateTime = pd.to_numeric(pd.to_datetime(events['DateTime']))/1e9 #convert ns to s
    DateTimeNext = DateTime.shift(periods=-1) # -1 because we want to find the gap until the NEXT one (ie. if THIS event failed)
    time_diff = (DateTimeNext-DateTime) #Find the difference in event times
    df_less15m = events[((time_diff>(30*60)) | (time_diff.isnull()))]#more than 30 min or when time_diff is null, as that is the last recorded event
    default_df = pd.DataFrame([[gr,EventID,'None']]) #Returned if no start events detected - filter out using Datetime == none
    if len(df_less15m.index)>0:
        return_df = pd.DataFrame(df_less15m[['CSN','EventID','DateTime']])
        #reset column names, this needs to be done otherwise it won't match the schema above. It works when we do .mean() etc
        #because that doesn't give a named column when we create the dataframe, whereas this does.
        return_df = return_df.T.reset_index(drop=True).T #this feels hacky
        return return_df
    else:
        return default_df








schema3 = StructType([
    StructField("CSN", StringType()),
    StructField("EventID", IntegerType()),
    StructField("DateTime", StringType())
])

@pandas_udf(schema3, functionType=PandasUDFType.GROUPED_MAP)
def failTimes(df):
    gr = df['CSN'].iloc[0]#used for the default returned dataframe if no data being returned
    EventID = 65533
    events = df[df['EventID']==EventID].sort_values(['DateTime'])
    DateTime = pd.to_numeric(pd.to_datetime(events['DateTime']))/1e9 #convert ns to s
    DateTimeNext = DateTime.shift(periods=-1) # -1 because we want to find the gap until the NEXT one (ie. if THIS event failed)
    time_diff = (DateTimeNext-DateTime)
    df_less15m = events[((time_diff<(30*60)) & (time_diff > 10))]#more than 10s, less than 30 mins
    default_df = pd.DataFrame([[gr,EventID,'None']])
    if len(df_less15m.index)>0:
        return_df = pd.DataFrame(df_less15m[['CSN','EventID','DateTime']])
        #reset column names, this needs to be done otherwise it won't match the schema above. It works when we do .mean() etc
        #because that doesn't give a named column when we create the dataframe, whereas this does.
        return_df = return_df.T.reset_index(drop=True).T #this feels hacky
        return return_df
    else:
        return default_df








schema5 = StructType([
    StructField("CSN", StringType()),
    StructField("EventID", IntegerType()),
    StructField("EventIDNext", IntegerType()),
    StructField("DateTime", StringType()),
    StructField("DateTimeInt", IntegerType()),
    StructField("DateTimeIntNext", IntegerType()),
    StructField("timeDiff", IntegerType())
])

@pandas_udf(schema5, functionType=PandasUDFType.GROUPED_MAP)
def removeDuplicates(df):
    df.rename(columns=lambda x: x.lstrip(), inplace = True)
    
    gr = df['CSN'].iloc[0]#used for the default returned dataframe if no data being returned
    EventID = 65533
    df['DateTimeInt'] = pd.to_numeric(pd.to_datetime(df['DateTime']))/1e9 #convert ns to s
    df = df.sort_values(['EventID','DateTimeInt'])

    df['DateTimeIntNext'] = df['DateTimeInt'].shift(periods=-1) # -1 because we want to find the gap until the NEXT one (ie. if THIS event failed)
    df['EventIDNext'] = df['EventID'].shift(periods=-1)
    
    df['timeDiff'] = (df['DateTimeIntNext']-df['DateTimeInt'])
    
   #Remove the rows with timeDiff between -1 and 1 and the same event ID
    dResult = df[((df.timeDiff < -1) | (df.timeDiff > 1)) | (df.EventID != df.EventIDNext)]
    
    #If empty array: i.e. only one event type happens, all at the same time, return the first row
    if dResult.empty:
        dResult = df.iloc[[0]]
    return dResult




schema4 = StructType([
StructField("CSN", StringType()),
StructField("EventID", IntegerType()),
StructField("DateTime", StringType()), 
StructField("TimeToStart", IntegerType()),
StructField("DateTimeInt", IntegerType()),
StructField("startIndex", IntegerType())
])

@pandas_udf(schema4, functionType=PandasUDFType.GROUPED_MAP)
def getClosestStartsandTimes(df):
    df.rename(columns=lambda x: x.lstrip(), inplace = True)
    gr = df['CSN'].iloc[0]#used for the default returned dataframe if no data being returned

    #Get time in seconds
    df['DateTimeInt'] = pd.to_numeric(pd.to_datetime(df['DateTime']))/1e9 #convert ns to s

    #Get Start Event times - startID can be changed for any event to anaylse train behaviour around that event
    startID = 65533
    startEvents = df[df['EventID']==startID]

    #if no start events for CSN, return empty array, otherwise perfrom calcs - reccommend that a check is performed after the calling of this function to check if any Datetime is listed as 'WRONG:
    if startEvents.empty:
        otherEvents = pd.DataFrame([[gr,65533,'WRONG',1,2,3]])
    else:#Otherwise...
        otherEvents = df[df['EventID']!=startID] #Remove the start events
        st = startEvents['DateTimeInt'].values #outputs a list
        ot = otherEvents['DateTimeInt'].values
        
        #put into a numpy array to make use of numpy functions
        stNp = np.array(st) #stNp is a list of the times of the start events for the time period - compare time of each event to these vals
        otNp = np.array(ot)
        
        #Get shape of df to find no of rows
        col = len(ot)
        ot = ot.reshape(col,1)

        #Add start times to each row of dataframe
        stList = stNp #Declaring the list to be appending stNp to
        for i in range(col-1):
            stList = np.vstack((stList,stNp))
        
        #compute the time diff for each start time
        timeDiff = ot - stList
        
        #Find the minimum time value, and the index which it occurs at
        x = np.argmin(abs(timeDiff),axis=1).reshape(col,1)
        otherEvents['startIndex'] = x
        
        #Store the minimum time in the correct rows
        timeDiffResult = []
        for i in range(col):
            val = (timeDiff[i,x[i]])
            timeDiffResult.append(val[0])

        #Make new numpy of timeDiffResult and get min - NOT USED ANY MORE
        minTime = np.argmin(abs(np.array(timeDiffResult)))

        #Make new col with the time ebtween said event and the nearest start Time
        otherEvents['TimeToStart'] = (timeDiffResult)


    return otherEvents




def timeOnOff(df):
    #Order the data by time (Seconds) and get time Diff
    df = df.sort_values(['DateTime']) #start events
    DateTime = pd.to_numeric(pd.to_datetime(df['DateTime']))/1e9 #convert ns to s
    DateTimeNext = DateTime.shift(periods=-1) # -1 because we want to find the gap until the NEXT one (ie. if THIS event failed)

    #get event ID diff - no sort
    EventID = df['EventID']
    EventIDNext = EventID.shift(periods=-1).values
    EventID = EventID.values

    #Remove last row
    EventID = EventID[:-1]
    EventIDNext = EventIDNext[:-1]

    EventIDDiff = abs(EventIDNext - EventID)
    #Good where start-shut-start-shut
    good = sum(EventIDDiff)
    bad = len(EventIDDiff) - good

    gr = df['CSN'].iloc[0]
    default_df = pd.DataFrame([[gr,good,bad]])
    return default_df


def shutTimes(df):
    gr = df['CSN'].iloc[0]#used for the default returned dataframe if no data being returned
    EventID = 65534
    events = df[df['EventID']==EventID].sort_values(['DateTime']) #start events
    DateTime = pd.to_numeric(pd.to_datetime(events['DateTime']))/1e9 #convert ns to s
    DateTimeNext = DateTime.shift(periods=-1) # -1 because we want to find the gap until the NEXT one (ie. if THIS event failed)
    time_diff = (DateTimeNext-DateTime)
    df_less15m = events[((time_diff>(10*60)) | (time_diff.isnull()))]#more than 30 min
    default_df = pd.DataFrame([[gr,EventID,'None']])
    if len(df_less15m.index)>0:
        return_df = pd.DataFrame(df_less15m[['CSN','EventID','DateTime']])
        #reset column names, this needs to be done otherwise it won't match the schema above. It works when we do .mean() etc
        #because that doesn't give a named column when we create the dataframe, whereas this does.
        return_df = return_df.T.reset_index(drop=True).T #this feels hacky
        return return_df
    else:
        return default_df


