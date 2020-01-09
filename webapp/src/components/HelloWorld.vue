<template>
  <v-container>
    <v-layout
      text-center
      wrap
    >

      <v-flex xs12>
        <h1 class="title1">
          Demo
        </h1>
        <p class="subheading font-weight-regular">
          Coppola, Palazzi, Vivace - January 2020
        </p>
      </v-flex>

  <v-flex xs12>

      <div class="my-2">
        <v-btn color="primary" depressed @click="toggled=!toggled">LDA</v-btn>
      </div>


      <div class="my-2">
        <v-btn color="primary" depressed>SENTIMENT ANALYSIS TEST</v-btn>
      </div>


  </v-flex>

  <v-flex xs12 v-if="toggled">
  <v-row>
        <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>        

  <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>
  
</v-row> <br>

 <v-row>

    <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>        <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>
  
</v-row> <br> <v-row>

        <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>        <v-card
    class="mx-auto"
    max-width="344"
  >
    <v-img
      src="https://cdn.vuetifyjs.com/images/cards/sunshine.jpg"
      height="200px"
    ></v-img>

    <v-card-title>
      Top western road trips
    </v-card-title>

    <v-card-subtitle>
      1,000 miles of wonder
    </v-card-subtitle>

    <v-card-actions>
      <v-btn
        color="purple"
        text
      >
        PYLAVDIS
      </v-btn>

    </v-card-actions>

  </v-card>
  
</v-row>
      </v-flex>

    </v-layout>
  </v-container>
</template>

<script>
export default {
  name: 'HelloWorld',

  data: () => ({
    toggled: false,
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