# US_Traffic
**Goal:** 

    To create a classification model that can predict the severity of the traffic accident based on various variables

### Phases
- Planning
- Acquire
- Prepare
- Explore
- Model
- Evaluate
- Conclusion


### Data Dictionary
| Attribute | Description                                         |                                                                                                                                                                                                                                 |
|:---------:|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     1     |                             ID                         |This is a unique identifier of the accident record.|
|     2     |                        Source                       | Indicates source of the accident report (i.e. the API which reported the accident.).                                                                                                                                            |
|     3     |                         TMC                         | A traffic accident may have a Traffic Message Channel (TMC) code which provides more detailed description of the event.                                                                                                         |
|     4     |                       Severity                      | Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay). |
|     5     |                      Start_Time                     | Shows start time of the accident in local time zone.                                                                                                                                                                            |
|     6     |                       End_Time                      | Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow was dismissed.                                                                                           |
|     7     |                      Start_Lat                      | Shows latitude in GPS coordinate of the start point.                                                                                                                                                                            |
|     8     |                      Start_Lng                      | Shows longitude in GPS coordinate of the start point.                                                                                                                                                                           |
|     9     |                       End_Lat                       | Shows latitude in GPS coordinate of the end point.                                                                                                                                                                              |
|     10    |                       End_Lng                       | Shows longitude in GPS coordinate of the end point.                                                                                                                                                                             |
|     11    |                     Distance(mi)                    | The length of the road extent affected by the accident.                                                                                                                                                                         |
|     12    |                     Description                     | Shows natural language description of the accident.                                                                                                                                                                             |
|     13    |                        Number                       | Shows the street number in address field.                                                                                                                                                                                       |
|     14    |                        Street                       | Shows the street name in address field.                                                                                                                                                                                         |
|     15    |                         Side                        | Shows the relative side of the street (Right/Left) in address field.                                                                                                                                                            |
|     16    |                         City                        | Shows the city in address field.                                                                                                                                                                                                |
|     17    |                        County                       | Shows the county in address field.                                                                                                                                                                                              |
|     18    |                        State                        | Shows the state in address field.                                                                                                                                                                                               |
|     19    |                       Zipcode                       | Shows the zipcode in address field.                                                                                                                                                                                             |
|     20    |                       Country                       | Shows the country in address field.                                                                                                                                                                                             |
|     21    |                       Timezone                      | Shows timezone based on the location of the accident (eastern, central, etc.).                                                                                                                                                  |
|     22    |                     Airport_Code                    | Denotes an airport-based weather station which is the closest one to location of the accident.                                                                                                                                  |
|     23    |                  Weather_Timestamp                  | Shows the time-stamp of weather observation record (in local time).                                                                                                                                                             |
|     24    |                    Temperature(F)                   | Shows the temperature (in Fahrenheit).                                                                                                                                                                                          |
|     25    |                    Wind_Chill(F)                    | Shows the wind chill (in Fahrenheit).                                                                                                                                                                                           |
|     26    |                     Humidity(%)                     | Shows the humidity (in percentage).                                                                                                                                                                                             |
|     27    |                     Pressure(in)                    | Shows the air pressure (in inches).                                                                                                                                                                                             |
|     28    |                    Visibility(mi)                   | Shows visibility (in miles).                                                                                                                                                                                                    |
|     29    |                    Wind_Direction                   | Shows wind direction.                                                                                                                                                                                                           |
|     30    |                   Wind_Speed(mph)                   | Shows wind speed (in miles per hour).                                                                                                                                                                                           |
|     31    |                  Precipitation(in)                  | Shows precipitation amount in inches, if there is any.                                                                                                                                                                          |
|     32    |                  Weather_Condition                  | Shows the weather condition (rain, snow, thunderstorm, fog, etc.)                                                                                                                                                               |
|     33    |                       Amenity                       | A POI annotation which indicates presence of amenity in a nearby location.                                                                                                                                                      |
|     34    |                         Bump                        | A POI annotation which indicates presence of speed bump or hump in a nearby location.                                                                                                                                           |
|     35    |                       Crossing                      | A POI annotation which indicates presence of crossing in a nearby location.                                                                                                                                                     |
|     36    |                       Give_Way                      | A POI annotation which indicates presence of give_way in a nearby location.                                                                                                                                                     |
|     37    |                       Junction                      | A POI annotation which indicates presence of junction in a nearby location.                                                                                                                                                     |
|     38    |                       No_Exit                       | A POI annotation which indicates presence of no_exit in a nearby location.                                                                                                                                                      |
|     39    |                       Railway                       | A POI annotation which indicates presence of railway in a nearby location.                                                                                                                                                      |
|     40    |                      Roundabout                     | A POI annotation which indicates presence of roundabout in a nearby location.                                                                                                                                                   |
|     41    |                       Station                       | A POI annotation which indicates presence of station in a nearby location.                                                                                                                                                      |
|     42    |                         Stop                        | A POI annotation which indicates presence of stop in a nearby location.                                                                                                                                                         |
|     43    |                   Traffic_Calming                   | A POI annotation which indicates presence of traffic_calming in a nearby location.                                                                                                                                              |
|     44    |                    Traffic_Signal                   | A POI annotation which indicates presence of traffic_signal in a nearby location.                                                                                                                                               |
|     45    |                     Turning_Loop                    | A POI annotation which indicates presence of turning_loop in a nearby location.                                                                                                                                                 |
|     46    |                    Sunrise_Sunset                   | Shows the period of day (i.e. day or night) based on sunrise/sunset.                                                                                                                                                            |
|     47    |                    Civil_Twilight                   | Shows the period of day (i.e. day or night) based on civil twilight.                                                                                                                                                            |
|     48    |                  Nautical_Twilight                  | Shows the period of day (i.e. day or night) based on nautical twilight.                                                                                                                                                         |
|     49    |                Astronomical_Twilight                | Shows the period of day (i.e. day or night) based on astronomical twilight.                                                                                                                                                     |

### Planning
- Look for project ideas
- Find the data for the project
- Explore data to identify the target variable
- Outline steps to proceed forward


### Acquire
-	Acquired data from Kaggle
-	Initially had 3 million observations
-	Filtered the data down by state
-	Chose only to work with California data
-	Saved the copy of the data on the local machine
-	Prepared a acquire.py module

### Prepare
-	Read the data
-	Dropped redundant columns
-	Normalized the column names
-	Dropped na
-	Extracted features like the day, month, and duration
-	Data conversion from boolean type to int type
-	Generated dummy variables for some of the categorical variables
-	Prepared a prepare.py module to store all of the functions


### Explore
-	Explored relationship between independent and dependent variables
-	Density plot of all of the numeric variables
-  Count plot of the categorical varibales
-  Plotted lat and long after filtering the data by different class of target var to identify the hotspots 



### Model
- Selected 15 features using selectKBest
- Used various classification algorithms to create models
- Trained these models on train data set


### Evaluate
- Models were evaluated using their accuracy
- RandomForest performed consistenly better than other models in all train, validate, and test data
- RandomFores is the final model



### Next steps
- Do the entire project again but using all of the 3 million observations
- Explore if KNN can be used without having to wait 10+ hours to train the model
- Explore other classification algorithms to see if they can imporve accuracy

