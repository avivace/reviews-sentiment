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
                        <v-tab @click="toggledlda=!toggledlda;toggledsent=false;toggledexploration=false" style="font-size: 1.2rem"> LDA </v-tab>

            <v-tab @click="toggledsent=!toggledsent;toggledlda=false;toggledexploration=false" style="font-size: 1.2rem"> SENTIMENT ANALYSIS </v-tab>
            <v-tab @click="toggledexploration=!toggledexploration;toggledlda=false;toggledsent=false" style="font-size: 1.2rem"> EXPLORATION</v-tab>
          </v-tabs>
          <v-flex lg12 xs12 v-if="toggledexploration">
            <v-row justify="center">
              <v-col cols="6" sm="6">
                <apexchart width="500" type="bar" :options="chartOptions" :series="series"></apexchart>
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
    toggledlda: true,
    toggledsent: false,
    value: 0,
    toggledexploration: false,
    chartOptions: {
      chart: {
        id: 'vuechart-example'
      },
      xaxis: {
        categories: [1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998]
      }
    },
    series: [{
      name: 'series-1',
      data: [30, 40, 35, 50, 49, 60, 70, 91]
    }],
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