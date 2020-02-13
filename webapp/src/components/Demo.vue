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
              Data Analytics project, Coppola, Palazzi, Vivace - January 2020
            </p>
          </v-flex>
          <v-tabs :grow="true" right="right" align-with-title background-color="transparent">
            <v-tab @click="toggledexploration=!toggledexploration;toggledlda=false;toggledsent=false" style="font-size: 1.2rem"> EXPLORATION</v-tab>
            <v-tab @click="toggledlda=!toggledlda;toggledsent=false;toggledexploration=false" style="font-size: 1.2rem"> LDA </v-tab>
            <v-tab @click="toggledsent=!toggledsent;toggledlda=false;toggledexploration=false" style="font-size: 1.2rem"> SENTIMENT ANALYSIS </v-tab>
          </v-tabs>
          <v-flex lg12 xs12 v-if="toggledexploration">
            <br><br>
            <v-row justify="center">
              <v-col cols="8" sm="8">
                <v-select v-model="selectedPlot" :items="plots" label="Select plot to show" solo></v-select>
              </v-col>
            </v-row>
            <v-row justify="center">
              <v-col cols="12" sm="12">
                <h3> </h3>
                <br>
                <h3> Verified VS Unverified average score </h3>
                Click on a bar to view the detailed distribution of reviews for that product
                <apexchart v-if="selectedPlot==0" @dataPointSelection="clickedPlot" width="100%" height="460px" :options="chartOptions" :series="series" type="bar"></apexchart>
                <br>
                <template v-if="selectedProduct!=null">
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
    plots: [{ value: 0, text: 'Verified VS Unverified study' }],
    reviewCountPlot: {
      dataLabels: {
        enabled: true
      },
      chart: {
        type: 'bar',
        height: 350,
        stacked: true,
        stackType: '100%'
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
        'decimalsInFloat': 1
      },
      chart: {},
      xaxis: {
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