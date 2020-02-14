<template>
  <v-app>
    <v-content>
      <v-container>
        <v-layout text-center wrap>
          <v-flex xs12>
            <h1 class="title1">
              Amazon Reviews Demo
            </h1>
            <p class="subheading font-weight-regular">
              Data Analytics project - Coppola, Palazzi, Vivace - January 2020 - <a style="text-decoration: none" href="https://github.com/avivace/reviews-sentiment">Source</a>
            </p>
          </v-flex>
          <v-tabs :grow="true" right="right" align-with-title background-color="transparent">
            <v-tab @click="toggledexploration=!toggledexploration;toggledlda=false;toggledsent=false" style="font-size: 1.2rem"> EXPLORATION</v-tab>
            <v-tab @click="toggledlda=!toggledlda;toggledsent=false;toggledexploration=false" style="font-size: 1.2rem"> LDA </v-tab>
            <!--<v-tab @click="toggledsent=!toggledsent;toggledlda=false;toggledexploration=false" style="font-size: 1.2rem"> SENTIMENT ANALYSIS </v-tab>-->
          </v-tabs>
          <v-flex lg12 xs12 v-if="toggledexploration">
            <br><br>
            <v-row justify="center">
              <v-col cols="10" sm="6">
                Select plot:
                <v-select v-model="selectedPlot" :items="plots" label="Select plot to show" solo></v-select>
              </v-col>
            </v-row>
            <v-row justify="center">
              <v-col cols="12" lg="9" sm="12" md="12">
                <h3> </h3>
                <br>
                <template v-if="selectedPlot==1">
                  <h3> Published reviews timeseries </h3>
                  <apexchart width="100%" height="460px" :options="timeseriesPlot" :series="timeseries"></apexchart>
                  <br><br>
                  <h3> Reviews per week day </h3>
                  <apexchart width="100%" height="600px" :options="weekdayPlot" :series="weekdaySeries"></apexchart>
                  <br><br>
                  <h3> Reviews per month </h3>
                  <apexchart width="100%" height="600px" :options="monthPlot" :series="monthSeries"></apexchart>
                </template>
                <template v-if="selectedPlot==0">
                  <h3> Verified VS Unverified average score </h3>
                  Click on a bar to view the detailed distribution of reviews for that product
                  <apexchart @dataPointSelection="clickedPlot" width="100%" height="460px" :options="chartOptions" :series="series" type="bar"></apexchart>
                </template>
                <br>
                <template v-if="selectedProduct!=null && selectedPlot==0">
                  <h3> Review count per rating for the product {{ this.chartOptions.xaxis.categories[selectedProduct] }} </h3>
                  <apexchart type="bar" height="250" :options="reviewCountPlot" :series="count[selectedProduct]"></apexchart>
                </template>
              </v-col>
            </v-row>
          </v-flex>
          <v-flex lg12 xs12 v-if="toggledsent">
            <v-row justify="center">
              <v-col cols="6" sm="6">
                <br>
                Write a custom review to see the evaluated sentiment:
                <v-text-field @change="compute" v-model="formText" label="Custom Review" outlined clearable counter hint="English only!"></v-text-field>
              </v-col>
            </v-row>
            <v-row justify="center">
              <center v-if="!errormsg"> {{ (value.toFixed(5) * 100).toFixed(2) }}% </center>
              <center v-else>
                <p class="red"> {{ errormsg }} <br></p> Did you check the backend is running correctly?
              </center>
            </v-row>
            <br>
            <v-btn @click=compute>compute </v-btn>
          </v-flex>
          <v-flex xs12 v-if="toggledlda">
            <v-container fluid>
              <v-row dense>
                <v-col v-for="object in lda" :key="object.nam" cols=4>
                  <v-card :key="object" class="mx-auto" max-width="344">
                    <v-img :src="object.code+'.jpg'" height="200px"></v-img>
                    <v-card-title>
                      {{ object.name }}
                    </v-card-title>
                    <v-card-subtitle>
                      {{ object.description }}
                    </v-card-subtitle>
                    <v-card-actions>
                      <v-btn :href="'lda_'+object.code+'.html'" target="_blank" text> pyLDAvis<v-icon right dark>mdi-open-in-new</v-icon>
                      </v-btn>
                    </v-card-actions>
                  </v-card>
                </v-col>
              </v-row>
            </v-container>
          </v-flex>
        </v-layout>
      </v-container>
    </v-content>
  </v-app>
</template>
<script>
export default {
  name: 'Demo',
  methods: {
    clickedPlot: function(a, b, c) {

      console.log("Selected product:", this.chartOptions.xaxis.categories[c.dataPointIndex])
      this.selectedProduct = c.dataPointIndex
    },
    compute: function() {
      this.errormsg = null
      let self = this
      this.$axios.get('http://localhost:5000/', {
          params: {
            text: this.formText
          }
        }).then(function(response) {
          self.value = response.data.positive
        })
        .catch(function(error) {
          self.errormsg = error + " on " + self.apiEndpoint
        })
    }
  },
  watch: {
    formText: function(val) {
      // Do things
      this.test = "A" + val
    },
  },
  data: () => ({
    apiEndpoint: "http://localhost:5000/",
    errormsg: null,
    formText: "",
    test: "aa",
    toggledlda: false,
    toggledsent: false,
    value: 0,
    toggledexploration: true,
    selectedPlot: 0,
    selectedProduct: null,
    plots: [{ value: 0, text: 'Verified VS Unverified study' },
      { value: 1, text: 'Time series' }
    ],
    timeseries: [{
        name: 'Verified',
        data: [{
            "x": 1359590400000,
            "y": 6695
          },
          {
            "x": 1362009600000,
            "y": 5573
          },
          {
            "x": 1364688000000,
            "y": 5695
          },
          {
            "x": 1367280000000,
            "y": 5396
          },
          {
            "x": 1369958400000,
            "y": 5983
          },
          {
            "x": 1372550400000,
            "y": 6143
          },
          {
            "x": 1375228800000,
            "y": 6922
          },
          {
            "x": 1377907200000,
            "y": 7201
          },
          {
            "x": 1380499200000,
            "y": 6128
          },
          {
            "x": 1383177600000,
            "y": 7165
          },
          {
            "x": 1385769600000,
            "y": 7365
          },
          {
            "x": 1388448000000,
            "y": 9100
          },
          {
            "x": 1391126400000,
            "y": 9902
          },
          {
            "x": 1393545600000,
            "y": 7986
          },
          {
            "x": 1396224000000,
            "y": 8851
          },
          {
            "x": 1398816000000,
            "y": 8218
          },
          {
            "x": 1401494400000,
            "y": 8150
          },
          {
            "x": 1404086400000,
            "y": 7870
          },
          {
            "x": 1406764800000,
            "y": 13108
          },
          {
            "x": 1409443200000,
            "y": 13672
          },
          {
            "x": 1412035200000,
            "y": 13701
          },
          {
            "x": 1414713600000,
            "y": 16113
          },
          {
            "x": 1417305600000,
            "y": 18793
          },
          {
            "x": 1419984000000,
            "y": 24933
          },
          {
            "x": 1422662400000,
            "y": 26000
          },
          {
            "x": 1425081600000,
            "y": 25609
          },
          {
            "x": 1427760000000,
            "y": 25188
          },
          {
            "x": 1430352000000,
            "y": 22130
          },
          {
            "x": 1433030400000,
            "y": 21208
          },
          {
            "x": 1435622400000,
            "y": 21738
          },
          {
            "x": 1438300800000,
            "y": 23350
          },
          {
            "x": 1440979200000,
            "y": 23587
          },
          {
            "x": 1443571200000,
            "y": 22651
          },
          {
            "x": 1446249600000,
            "y": 24873
          },
          {
            "x": 1448841600000,
            "y": 23685
          },
          {
            "x": 1451520000000,
            "y": 23841
          },
          {
            "x": 1454198400000,
            "y": 26236
          },
          {
            "x": 1456704000000,
            "y": 22863
          },
          {
            "x": 1459382400000,
            "y": 25195
          },
          {
            "x": 1461974400000,
            "y": 23633
          },
          {
            "x": 1464652800000,
            "y": 23699
          },
          {
            "x": 1467244800000,
            "y": 22660
          },
          {
            "x": 1469923200000,
            "y": 23708
          },
          {
            "x": 1472601600000,
            "y": 24498
          },
          {
            "x": 1475193600000,
            "y": 21666
          },
          {
            "x": 1477872000000,
            "y": 18863
          },
          {
            "x": 1480464000000,
            "y": 16909
          },
          {
            "x": 1483142400000,
            "y": 19576
          },
          {
            "x": 1485820800000,
            "y": 19031
          },
          {
            "x": 1488240000000,
            "y": 13601
          },
          {
            "x": 1490918400000,
            "y": 14918
          },
          {
            "x": 1493510400000,
            "y": 12096
          },
          {
            "x": 1496188800000,
            "y": 10515
          },
          {
            "x": 1498780800000,
            "y": 9543
          },
          {
            "x": 1501459200000,
            "y": 9268
          },
          {
            "x": 1504137600000,
            "y": 8965
          },
          {
            "x": 1506729600000,
            "y": 7517
          },
          {
            "x": 1509408000000,
            "y": 6884
          },
          {
            "x": 1512000000000,
            "y": 6789
          },
          {
            "x": 1514678400000,
            "y": 6029
          }
        ]
      }, {
        name: 'Unverified',
        data: [{
            "x": 1359590400000,
            "y": 694
          },
          {
            "x": 1362009600000,
            "y": 548
          },
          {
            "x": 1364688000000,
            "y": 537
          },
          {
            "x": 1367280000000,
            "y": 501
          },
          {
            "x": 1369958400000,
            "y": 607
          },
          {
            "x": 1372550400000,
            "y": 587
          },
          {
            "x": 1375228800000,
            "y": 663
          },
          {
            "x": 1377907200000,
            "y": 583
          },
          {
            "x": 1380499200000,
            "y": 532
          },
          {
            "x": 1383177600000,
            "y": 665
          },
          {
            "x": 1385769600000,
            "y": 605
          },
          {
            "x": 1388448000000,
            "y": 732
          },
          {
            "x": 1391126400000,
            "y": 795
          },
          {
            "x": 1393545600000,
            "y": 682
          },
          {
            "x": 1396224000000,
            "y": 824
          },
          {
            "x": 1398816000000,
            "y": 868
          },
          {
            "x": 1401494400000,
            "y": 985
          },
          {
            "x": 1404086400000,
            "y": 1847
          },
          {
            "x": 1406764800000,
            "y": 4862
          },
          {
            "x": 1409443200000,
            "y": 5429
          },
          {
            "x": 1412035200000,
            "y": 5189
          },
          {
            "x": 1414713600000,
            "y": 5991
          },
          {
            "x": 1417305600000,
            "y": 3784
          },
          {
            "x": 1419984000000,
            "y": 2402
          },
          {
            "x": 1422662400000,
            "y": 2805
          },
          {
            "x": 1425081600000,
            "y": 2738
          },
          {
            "x": 1427760000000,
            "y": 3529
          },
          {
            "x": 1430352000000,
            "y": 3527
          },
          {
            "x": 1433030400000,
            "y": 3331
          },
          {
            "x": 1435622400000,
            "y": 2781
          },
          {
            "x": 1438300800000,
            "y": 2593
          },
          {
            "x": 1440979200000,
            "y": 2829
          },
          {
            "x": 1443571200000,
            "y": 3360
          },
          {
            "x": 1446249600000,
            "y": 3333
          },
          {
            "x": 1448841600000,
            "y": 4304
          },
          {
            "x": 1451520000000,
            "y": 4071
          },
          {
            "x": 1454198400000,
            "y": 4149
          },
          {
            "x": 1456704000000,
            "y": 3320
          },
          {
            "x": 1459382400000,
            "y": 4051
          },
          {
            "x": 1461974400000,
            "y": 4759
          },
          {
            "x": 1464652800000,
            "y": 4484
          },
          {
            "x": 1467244800000,
            "y": 6143
          },
          {
            "x": 1469923200000,
            "y": 4830
          },
          {
            "x": 1472601600000,
            "y": 3115
          },
          {
            "x": 1475193600000,
            "y": 2665
          },
          {
            "x": 1477872000000,
            "y": 1869
          },
          {
            "x": 1480464000000,
            "y": 1260
          },
          {
            "x": 1483142400000,
            "y": 1396
          },
          {
            "x": 1485820800000,
            "y": 1132
          },
          {
            "x": 1488240000000,
            "y": 723
          },
          {
            "x": 1490918400000,
            "y": 800
          },
          {
            "x": 1493510400000,
            "y": 746
          },
          {
            "x": 1496188800000,
            "y": 666
          },
          {
            "x": 1498780800000,
            "y": 666
          },
          {
            "x": 1501459200000,
            "y": 833
          },
          {
            "x": 1504137600000,
            "y": 679
          },
          {
            "x": 1506729600000,
            "y": 500
          },
          {
            "x": 1509408000000,
            "y": 459
          },
          {
            "x": 1512000000000,
            "y": 407
          },
          {
            "x": 1514678400000,
            "y": 403
          }
        ]
      },
      {
        name: 'Total',
        data: [{
            "x": 1359590400000,
            "y": 7389
          },
          {
            "x": 1362009600000,
            "y": 6121
          },
          {
            "x": 1364688000000,
            "y": 6232
          },
          {
            "x": 1367280000000,
            "y": 5897
          },
          {
            "x": 1369958400000,
            "y": 6590
          },
          {
            "x": 1372550400000,
            "y": 6730
          },
          {
            "x": 1375228800000,
            "y": 7585
          },
          {
            "x": 1377907200000,
            "y": 7784
          },
          {
            "x": 1380499200000,
            "y": 6660
          },
          {
            "x": 1383177600000,
            "y": 7830
          },
          {
            "x": 1385769600000,
            "y": 7970
          },
          {
            "x": 1388448000000,
            "y": 9832
          },
          {
            "x": 1391126400000,
            "y": 10697
          },
          {
            "x": 1393545600000,
            "y": 8668
          },
          {
            "x": 1396224000000,
            "y": 9675
          },
          {
            "x": 1398816000000,
            "y": 9086
          },
          {
            "x": 1401494400000,
            "y": 9135
          },
          {
            "x": 1404086400000,
            "y": 9717
          },
          {
            "x": 1406764800000,
            "y": 17970
          },
          {
            "x": 1409443200000,
            "y": 19101
          },
          {
            "x": 1412035200000,
            "y": 18890
          },
          {
            "x": 1414713600000,
            "y": 22104
          },
          {
            "x": 1417305600000,
            "y": 22577
          },
          {
            "x": 1419984000000,
            "y": 27335
          },
          {
            "x": 1422662400000,
            "y": 28805
          },
          {
            "x": 1425081600000,
            "y": 28347
          },
          {
            "x": 1427760000000,
            "y": 28717
          },
          {
            "x": 1430352000000,
            "y": 25657
          },
          {
            "x": 1433030400000,
            "y": 24539
          },
          {
            "x": 1435622400000,
            "y": 24519
          },
          {
            "x": 1438300800000,
            "y": 25943
          },
          {
            "x": 1440979200000,
            "y": 26416
          },
          {
            "x": 1443571200000,
            "y": 26011
          },
          {
            "x": 1446249600000,
            "y": 28206
          },
          {
            "x": 1448841600000,
            "y": 27989
          },
          {
            "x": 1451520000000,
            "y": 27912
          },
          {
            "x": 1454198400000,
            "y": 30385
          },
          {
            "x": 1456704000000,
            "y": 26183
          },
          {
            "x": 1459382400000,
            "y": 29246
          },
          {
            "x": 1461974400000,
            "y": 28392
          },
          {
            "x": 1464652800000,
            "y": 28183
          },
          {
            "x": 1467244800000,
            "y": 28803
          },
          {
            "x": 1469923200000,
            "y": 28538
          },
          {
            "x": 1472601600000,
            "y": 27613
          },
          {
            "x": 1475193600000,
            "y": 24331
          },
          {
            "x": 1477872000000,
            "y": 20732
          },
          {
            "x": 1480464000000,
            "y": 18169
          },
          {
            "x": 1483142400000,
            "y": 20972
          },
          {
            "x": 1485820800000,
            "y": 20163
          },
          {
            "x": 1488240000000,
            "y": 14324
          },
          {
            "x": 1490918400000,
            "y": 15718
          },
          {
            "x": 1493510400000,
            "y": 12842
          },
          {
            "x": 1496188800000,
            "y": 11181
          },
          {
            "x": 1498780800000,
            "y": 10209
          },
          {
            "x": 1501459200000,
            "y": 10101
          },
          {
            "x": 1504137600000,
            "y": 9644
          },
          {
            "x": 1506729600000,
            "y": 8017
          },
          {
            "x": 1509408000000,
            "y": 7343
          },
          {
            "x": 1512000000000,
            "y": 7196
          },
          {
            "x": 1514678400000,
            "y": 6432
          }
        ]
      }
    ],
    timeseriesPlot: {
      fill: {
        type: 'gradient',
        gradient: {
          shadeIntensity: 0.5,
          inverseColors: false,
          opacityFrom: 1,
          opacityTo: 0.9,
          stops: [0, 90, 100]
        },
      },
      legend: {
        fontSize: '18px',
      },
      chart: {

        toolbar: {
          show: true,


        },
        height: 350,
      },
      xaxis: {
        type: 'datetime'
      }
    },
    weekdayPlot: {
      dataLabels: {
        enabled: true
      },
      chart: {
        type: 'bar',
        height: 350,
        stacked: true,

        toolbar: {
          show: false
        }
      },
      plotOptions: {
        bar: {
          horizontal: false,
        },
      },
      xaxis: {
        categories: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
      },
      yaxis: {
        labels: {
          formatter: function(val) {
            return val / 1000 + 'K'
          }
        }
      }
    },
    monthPlot: {
      dataLabels: {
        enabled: true
      },
      chart: {
        type: 'bar',
        height: 350,
        stacked: true,
        //stackType: '100%',
        toolbar: {
          show: false
        }
      },
      plotOptions: {
        bar: {
          horizontal: false,
        },
      },
      xaxis: {
        categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
      },
      yaxis: {
        labels: {
          formatter: function(val) {
            return val / 1000 + 'K'
          }
        }
      }
    },
    weekdaySeries: [{ name: "1", data: [12638, 12838, 12481, 12010, 11258, 10081, 10200] },
      { name: "2", data: [8944, 8997, 8757, 8461, 8025, 6855, 7127] },
      { name: "3", data: [15531, 15521, 15275, 14627, 13721, 11695, 11843] },
      { name: "4", data: [29982, 29097, 28286, 27227, 25107, 22009, 22638] },
      { name: "5", data: [114477, 110927, 108314, 104506, 96903, 84637, 86659] }
    ],
    monthSeries: [{ name: "1", data: [7308, 6308, 6868, 6610, 6357, 6355, 7339, 7120, 6647, 6705, 6617, 7272] },
      { name: "2", data: [5407, 4384, 4760, 4569, 4479, 4472, 5009, 5111, 4417, 4672, 4559, 5327] },
      { name: "3", data: [9458, 7819, 8436, 7758, 7568, 7643, 8591, 8349, 7515, 7876, 8126, 9074] },
      { name: "4", data: [17668, 14918, 15667, 14730, 14316, 14569, 16027, 15796, 14338, 14902, 14670, 16745] },
      { name: "5", data: [67782, 58541, 62850, 56459, 54246, 54163, 60865, 60382, 55466, 56339, 55773, 63557] }
    ],
    reviewCountPlot: {
      dataLabels: {
        enabled: true
      },
      chart: {
        type: 'bar',
        height: 350,
        stacked: true,
        stackType: '100%',
        toolbar: {
          show: false
        }
      },
      plotOptions: {
        bar: {
          horizontal: true,
        },
      },
      stroke: {
        width: 1,
        colors: ['#fff']
      },

      xaxis: {
        categories: ['Verified', 'Unverified']
      },
      fill: {
        opacity: 1
      },
      legend: {
        position: 'top',
        horizontalAlign: 'left',
        offsetX: 40
      }
    },
    chartOptions: {

      dataLabels: {
        enabled: false
      },
      yaxis: {
        labels: {
          style: {
            fontSize: '18px'
          },

        },
        'decimalsInFloat': 1
      },
      chart: {
        toolbar: {
          show: false
        }
      },
      xaxis: {
        labels: {
          style: {
            fontSize: '18px'
          },

        },
        'decimalsInFloat': 1,
        categories: ["B005NF5NTK", "B0092KJ9BU", "B00AANQLRI", "B00BT8L2MW", "B00D856NOG", "B00G7UY3EG", "B00IGISUTG", "B00M51DDT2", "B00M6QODH2", "B00MQSMDYU", "B00MXWFUQC", "B00P7N0320", "B00QN1T6NM", "B00UCZGS6S", "B00UH3L82Y", "B00VH88CJ0", "B00X5RV14Y", "B014EB532U", "B018JW3EOY", "B019PV2I3G"]
      },
    },

    // Data source: ver_counts.json (generated by data_exploration.count_reviews)
    count: [
      [{
          "data": [
            47,
            5
          ],
          "name": 1
        },
        {
          "data": [
            32,
            2
          ],
          "name": 2
        },
        {
          "data": [
            79,
            5
          ],
          "name": 3
        },
        {
          "data": [
            231,
            9
          ],
          "name": 4
        },
        {
          "data": [
            1189,
            35
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            64,
            8
          ],
          "name": 1
        },
        {
          "data": [
            72,
            9
          ],
          "name": 2
        },
        {
          "data": [
            90,
            17
          ],
          "name": 3
        },
        {
          "data": [
            155,
            28
          ],
          "name": 4
        },
        {
          "data": [
            580,
            41
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            22,
            3
          ],
          "name": 1
        },
        {
          "data": [
            14,
            1
          ],
          "name": 2
        },
        {
          "data": [
            37,
            4
          ],
          "name": 3
        },
        {
          "data": [
            141,
            2
          ],
          "name": 4
        },
        {
          "data": [
            774,
            25
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            35,
            3
          ],
          "name": 1
        },
        {
          "data": [
            31,
            3
          ],
          "name": 2
        },
        {
          "data": [
            35,
            1
          ],
          "name": 3
        },
        {
          "data": [
            93,
            3
          ],
          "name": 4
        },
        {
          "data": [
            800,
            24
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            11,
            2
          ],
          "name": 1
        },
        {
          "data": [
            14,
            1
          ],
          "name": 2
        },
        {
          "data": [
            38,
            1
          ],
          "name": 3
        },
        {
          "data": [
            139,
            12
          ],
          "name": 4
        },
        {
          "data": [
            773,
            50
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            132,
            11
          ],
          "name": 1
        },
        {
          "data": [
            41,
            0
          ],
          "name": 2
        },
        {
          "data": [
            48,
            1
          ],
          "name": 3
        },
        {
          "data": [
            120,
            11
          ],
          "name": 4
        },
        {
          "data": [
            942,
            71
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            40,
            2
          ],
          "name": 1
        },
        {
          "data": [
            26,
            0
          ],
          "name": 2
        },
        {
          "data": [
            36,
            0
          ],
          "name": 3
        },
        {
          "data": [
            103,
            2
          ],
          "name": 4
        },
        {
          "data": [
            892,
            18
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            40,
            5
          ],
          "name": 1
        },
        {
          "data": [
            21,
            6
          ],
          "name": 2
        },
        {
          "data": [
            41,
            3
          ],
          "name": 3
        },
        {
          "data": [
            120,
            8
          ],
          "name": 4
        },
        {
          "data": [
            954,
            69
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            33,
            7
          ],
          "name": 1
        },
        {
          "data": [
            24,
            7
          ],
          "name": 2
        },
        {
          "data": [
            38,
            5
          ],
          "name": 3
        },
        {
          "data": [
            89,
            26
          ],
          "name": 4
        },
        {
          "data": [
            592,
            156
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            39,
            2
          ],
          "name": 1
        },
        {
          "data": [
            19,
            3
          ],
          "name": 2
        },
        {
          "data": [
            41,
            2
          ],
          "name": 3
        },
        {
          "data": [
            134,
            10
          ],
          "name": 4
        },
        {
          "data": [
            1006,
            84
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            97,
            7
          ],
          "name": 1
        },
        {
          "data": [
            70,
            3
          ],
          "name": 2
        },
        {
          "data": [
            83,
            2
          ],
          "name": 3
        },
        {
          "data": [
            148,
            14
          ],
          "name": 4
        },
        {
          "data": [
            484,
            70
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            24,
            3
          ],
          "name": 1
        },
        {
          "data": [
            19,
            0
          ],
          "name": 2
        },
        {
          "data": [
            41,
            1
          ],
          "name": 3
        },
        {
          "data": [
            154,
            13
          ],
          "name": 4
        },
        {
          "data": [
            1109,
            149
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            62,
            4
          ],
          "name": 1
        },
        {
          "data": [
            46,
            2
          ],
          "name": 2
        },
        {
          "data": [
            67,
            6
          ],
          "name": 3
        },
        {
          "data": [
            162,
            22
          ],
          "name": 4
        },
        {
          "data": [
            754,
            102
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            75,
            8
          ],
          "name": 1
        },
        {
          "data": [
            46,
            5
          ],
          "name": 2
        },
        {
          "data": [
            70,
            3
          ],
          "name": 3
        },
        {
          "data": [
            161,
            8
          ],
          "name": 4
        },
        {
          "data": [
            609,
            23
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            152,
            6
          ],
          "name": 1
        },
        {
          "data": [
            81,
            2
          ],
          "name": 2
        },
        {
          "data": [
            121,
            4
          ],
          "name": 3
        },
        {
          "data": [
            137,
            10
          ],
          "name": 4
        },
        {
          "data": [
            467,
            26
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            25,
            2
          ],
          "name": 1
        },
        {
          "data": [
            20,
            1
          ],
          "name": 2
        },
        {
          "data": [
            29,
            4
          ],
          "name": 3
        },
        {
          "data": [
            169,
            8
          ],
          "name": 4
        },
        {
          "data": [
            1355,
            69
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            14,
            5
          ],
          "name": 1
        },
        {
          "data": [
            11,
            1
          ],
          "name": 2
        },
        {
          "data": [
            31,
            3
          ],
          "name": 3
        },
        {
          "data": [
            120,
            18
          ],
          "name": 4
        },
        {
          "data": [
            1133,
            177
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            80,
            4
          ],
          "name": 1
        },
        {
          "data": [
            45,
            1
          ],
          "name": 2
        },
        {
          "data": [
            60,
            0
          ],
          "name": 3
        },
        {
          "data": [
            121,
            5
          ],
          "name": 4
        },
        {
          "data": [
            668,
            17
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            23,
            3
          ],
          "name": 1
        },
        {
          "data": [
            18,
            0
          ],
          "name": 2
        },
        {
          "data": [
            41,
            1
          ],
          "name": 3
        },
        {
          "data": [
            156,
            12
          ],
          "name": 4
        },
        {
          "data": [
            1111,
            141
          ],
          "name": 5
        }
      ],
      [{
          "data": [
            37,
            5
          ],
          "name": 1
        },
        {
          "data": [
            39,
            3
          ],
          "name": 2
        },
        {
          "data": [
            36,
            0
          ],
          "name": 3
        },
        {
          "data": [
            102,
            4
          ],
          "name": 4
        },
        {
          "data": [
            1216,
            66
          ],
          "name": 5
        }
      ]
    ],
    // Data source: ver_unver.json (generated by data_exploration.top_50_products_verified_unverified_both)
    series: [

      {
        "name": "verified",
        "data": [4.573510773130545, 4.160249739854319, 4.65080971659919, 4.601609657947686, 4.691282051282052, 4.324240062353858, 4.62351868732908, 4.63860544217687, 4.524484536082475, 4.653753026634383, 3.9659863945578233, 4.711210096510765, 4.374885426214482, 4.231009365244537, 3.7160751565762005, 4.757822277847309, 4.792971734148205, 4.285420944558521, 4.715344699777613, 4.693006993006993]
      },
      {
        "name": "unverified",
        "data": [4.196428571428571, 3.825242718446602, 4.285714285714286, 4.235294117647059, 4.621212121212121, 4.3936170212765955, 4.545454545454546, 4.428571428571429, 4.577114427860696, 4.693069306930693, 4.427083333333333, 4.837349397590361, 4.588235294117647, 3.702127659574468, 4.0, 4.678571428571429, 4.769607843137255, 4.111111111111111, 4.834394904458598, 4.576923076923077]
      },
      {
        "name": "all",
        "data": [4.560587515299877, 4.12781954887218, 4.638318670576735, 4.589494163424124, 4.686839577329491, 4.328976034858388, 4.621983914209116, 4.623520126282557, 4.535312180143296, 4.656716417910448, 4.011247443762781, 4.725049570389953, 4.398533007334963, 4.2063492063492065, 3.7296222664015906, 4.7538644470868014, 4.789821546596166, 4.280719280719281, 4.727755644090306, 4.687002652519894]
      }
    ],

    lda: [{
        //
        name: "TaoTronics Car Phone Mount Holder, Windshield /",
        description: "Dashboard Universal Car Mobile Phone cradle for iOS / Android Smartphone and More",
        code: "B00MXWFUQC",
      },
      {
        name: "ArmorSuit MilitaryShield Max Coverage Screen Protector for Apple Watch",
        description: "42mm (Series 3 / 2 / 1 Compatible) [2 Pack] - Anti-Bubble HD Clear Film",
        code: "B00UH3L82Y",
      },
      {
        name: "Portable Charger Anker PowerCore 20100mAh",
        description: "Ultra High Capacity Power Bank with 4.8A Output and PowerIQ Technology.",
        code: "B00X5RV14Y",
      },
      {
        //
        name: "Anker 24W Dual USB Car Charger",
        description: "PowerDrive 2 for iPhone X / 8/7 / 6s / Plus, iPad Pro/Air 2 / Mini, Note 5/4, LG, Nexus, HTC, and More",
        code: "B00VH88CJ0",
      },
      {
        name: "Anker Astro E1 5200mAh Candy bar-Sized Ultra Compact Portable Charger",
        description: "(External Battery Power Bank) with High-Speed Charging PowerIQ Technology",
        code: "B018JW3EOY",
      },
      {
        //
        name: "Plantronics Voyager Legend Wireless Bluetooth Headset",
        description: "Compatible with iPhone, Android, and Other Leading Smartphones - Black",
        code: "B0092KJ9BU",
      },
      {
        //
        name: "Portable chargers",
        description: "Aggregated LDA",
        code: "PORTABLECHARGERS",
      }
    ],
    ecosystem: []
  }),
};
</script>
<style>
@import url('https://fonts.googleapis.com/css?family=Barlow');

.v-application .headline {
  font-family: 'Barlow' !important;
}

.v-application {
  font-family: 'Barlow' !important;
}

body {
  font-family: 'Barlow' !important;
  font-size: 20px;
}

html {
  font-family: 'Barlow' !important;
}

.headline {
  font-family: 'Barlow' !important;
}

.title {
  font-family: 'Barlow' !important;
}


.title1 {
  font-family: 'Barlow' !important;
  font-weight: 600;
}


h1 {
  font-family: 'Barlow' !important;
}



.apexcharts-legend-text {
  font-family: 'Barlow';
}

img {

  -webkit-box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
  -moz-box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
  box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
}

.slidecontainer {
  font-size: 24px;
}

input[type="number"] {
  width: 75px;
}

[class^="apex"],
[class*=" apex"] {
  font-size: 12px;
}
</style>