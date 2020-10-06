#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

using namespace std;

#define NUMBER_HOUR 24
#define NUMBER_SIGN 24
#define MINUTE_MEA 5
#define MINUTES_IN_HOUR 60

int measures = NUMBER_HOUR * (MINUTES_IN_HOUR / MINUTE_MEA);

int main() {
    //create random numbers
    srand(time(NULL));

    // file pointer 
    fstream fout; 

    // opens an existing csv file or creates a new file. 
    fout.open("data.csv", ios::out | ios::app); 

    int count = 0;

    for (int i = 0; i < measures; i++) {
        string tobe_time = to_string(count);
        string hour, minute, time;
        for (int j = 0; j < NUMBER_SIGN; j++) {
            if (count < 100) {
                hour = "00";
                minute = tobe_time;
                time = hour + ":" + minute;
            } else if (count < 1000) {
                hour = "0" + tobe_time.substr(0, 1);
                minute = tobe_time.substr(1);
                time = hour + ":" + minute;
            } else if (count < 10000){
                hour = tobe_time.substr(0, 2);
                minute = tobe_time.substr(2);
                time = hour + ":" + minute;
            } else {
                cout << "error" << endl;
                break;
            }
            int n = rand() % 100;

            fout << time << ","
            << j << ","
            << n << "\n";
        }
        if (minute == "55") {
            count += 45;
        } else {
            count += MINUTE_MEA;   
        }
    }

    cout << "create csv successfully" << endl;

    return 0;
}