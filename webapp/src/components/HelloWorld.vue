<template>
    <v-app>
      <v-app-bar class="navtitle"
      app
      color="red darken-4"
      dark
    >
Amazon Reviews Sentiment Analysis
<v-spacer></v-spacer>
    <v-tabs

right="right"
align-with-title
          background-color="transparent"
    >
      <v-tab  @click="toggledlda=!toggledlda;toggledsent=false" style="font-size: 1rem"> LDA </v-tab>
      <v-tab style="font-size: 1rem" @click="toggledsent=!toggledsent;toggledlda=false"> SENTIMENT ANALYSIS </v-tab>

      </v-tabs>
    </v-app-bar>
    <v-content>

  <v-container>
    <v-layout
      text-center
      wrap
    >

      <v-flex xs12>
        <br><br>
    
        <h1 class="title1">
          Demo
        </h1>
        <p class="subheading font-weight-regular">
          Coppola, Palazzi, Vivace - January 2020
        </p>
      </v-flex>



  <v-flex lg12 xs12 v-if="toggledsent">
    <v-row justify="center">
            <v-col cols="6" sm="6" >
          <v-text-field
            v-model="formText"
            label="Custom Review"
            outlined
            clearable
            counter
            hint="English only!"
          ></v-text-field>
        </v-col>
        
      </v-row>
      <v-row justify="center"> <center> {{ (value.toFixed(5) * 100).toFixed(2) }}% </center> </v-row>
      <br>
      <v-btn @click=compute
      >compute </v-btn>
    
  </v-flex>

  <v-flex xs12 v-if="toggledlda">
  <v-container fluid>
      <v-row dense>
        <v-col
          v-for="object in lda"
          :key="object.nam"
          cols=4
        >

        <v-card
        :key="object"
    class="mx-auto"
    max-width="344"
  >
    <v-img
      :src="object.code+'.jpg'"
      height="200px"
    ></v-img>

    <v-card-title>
      {{ object.name }}
    </v-card-title>

    <v-card-subtitle>
      {{ object.description }}
    </v-card-subtitle>

    <v-card-actions>
      <v-btn

        :href="'lda_'+object.code+'.html'"
        target="_blank"
        
        text
      >      pyLDAvis<v-icon right dark>mdi-open-in-new</v-icon>
        
 
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
  name: 'HelloWorld',
  methods: {
    compute: function(){
      
      this.$axios.get('http://localhost:5000/', {params: {
        text: this.formText
      }}).then(function (response) {
        self.value=response.data.positive
      })
    }
  },
  watch: {
    formText: function (val) {
      // Do things
      this.test = "A" + val
      var self = this;
      this.$axios.get('http://localhost:5000/', {params: {
        text: self.formText
      }}).then(function (response) {
        self.value=response.data.positive
      })
    
  },
},
  data: () => ({
    formText: "",
    test: "aa",
    toggledlda: false,
    value: 0,
    toggledsent: false,
    lda: [
    {
      name:"TaoTronics Car Phone Mount Holder, Windshield /",
      description:"Dashboard Universal Car Mobile Phone cradle for iOS / Android Smartphone and More",
      code:"B00MXWFUQC",
    },
    {
      name:"Samsung Qi Certified Wireless Charging Pad",
      description:"with 2A Wall Charger- Supports wireless charging on Qi compatible smartphones.",
      code:"B00UCZGS6S",
    },
    {
      name:"Portable Charger Anker PowerCore 20100mAh",
      description:"Ultra High Capacity Power Bank with 4.8A Output and PowerIQ Technology.",
      code:"B00X5RV14Y",
    },
    {
      name:"Anker 24W Dual USB Car Charger",
      description:"PowerDrive 2 for iPhone X / 8/7 / 6s / Plus, iPad Pro/Air 2 / Mini, Note 5/4, LG, Nexus, HTC, and More",
      code:"B00VH88CJ0",
    },
    {
      name:"Anker PowerCore+ mini 3350mAh Lipstick-Sized Portable Charger",
      description:"(3rd Generation, Premium Aluminum Power Bank) One of the Most Compact External Batteries",
      code:"B005NF5NTK",
    },
    {
      name:"Plantronics Voyager Legend Wireless Bluetooth Headset",
      description:"Compatible with iPhone, Android, and Other Leading Smartphones - Black",
      code:"B0092KJ9BU",
    }],
    ecosystem: []
  }),
};
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Barlow');

.v-application .headline{
  font-family: 'Barlow' !important;
}

.v-application{
  font-family: 'Barlow' !important;
}

body{
  font-family: 'Barlow' !important;
  font-size:20px;
}

html{
  font-family: 'Barlow' !important;
}

.headline{
  font-family: 'Barlow' !important;
}

.title{
  font-family: 'Barlow' !important;
}


.title1{
  font-family: 'Barlow' !important;
  font-weight: 600;
}


h1{
  font-family: 'Barlow' !important;
}



.apexcharts-legend-text {
  font-family: 'Barlow';
} 
img{
  
  -webkit-box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
  -moz-box-shadow: rgba(0,0,0,0.5) 0 2px 5px;
  box-shadow: rgba(0, 0, 0, 0.5) 0 2px 5px;
}
.slidecontainer{
  font-size:24px;
}
input[type="number"] {
   width:75px;
}
[class^="apex"], [class*=" apex"] { 
font-size:12px;
 }
</style>