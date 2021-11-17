async function getDetail(anchor) {
  const id = anchor.id
  console.log(id)
  const url = 'http://localhost:8000/area/' + id 
  const response = await fetch(url, {
    method: 'GET', 
    mode: 'cors',
    cache: 'no-cache', 
    credentials: 'same-origin', 
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin' : "*"
    },
    redirect: 'follow', 
    referrerPolicy: 'no-referrer',
  });
  return response.json(); 
}

const getDetailArea = async  (areaImage) => {
  console.log("AREA IMAGE" , areaImage)
  const unet_url = 'http://localhost:8000/unet/'
  const response = await fetch(unet_url, {
    method: 'POST',
    mode: 'cors',
    cache: 'no-cache',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin' : "*"
    },
    redirect: 'follow',
    referrerPolicy: 'no-referrer',
    body: JSON.stringify({
      "hashImage": areaImage,
    })
  })
  return response.json()
}


window.onload = function() {
  var anchors = document.getElementsByClassName('btn-get');
  console.log(anchors)
  for(var i = 0; i < anchors.length; i++) {
      const anchor = anchors[i];
      console.log(anchor)
      anchor.onclick = function() {
          document.querySelector('.bg-modal').style.display = "flex";
          var cur = document.getElementById('cur')
          var pre = document.getElementById('pre')
          var cur_mask = document.getElementById('cur_mask')
          var pre_mask = document.getElementById('pre_mask')
          var info_cur = document.getElementById('info_1')
          var info_pre = document.getElementById('info_2')
          
          // alert(anchor.id)
          getDetail(anchor)
              .then(data => {
                  const areaData = data.data[0]

                  cur.setAttribute(
                    'src', `data:image/png;base64,${areaData.current_images}`
                  );

                  pre.setAttribute(
                      'src', `data:image/png;base64,${areaData.previous_images}`
                  );
                  console.log(areaData)
                  getDetailArea(areaData.current_images)
                    .then(data => {
                      console.log("PREDICT" , data.data[0])
                      const predict = data.data[0]
                      cur_mask.setAttribute(
                        'src', `data:image/png;base64,${predict.image}`
                    );
                      info_cur.innerHTML = `Building : ${predict.building} , Area : ${predict.area}`
                    })

                  getDetailArea(areaData.previous_images)
                  .then(data => {
                    const predict = data.data[0]
                    console.log("PREDICT" , data.data[0])
                    pre_mask.setAttribute(
                      'src', `data:image/png;base64,${predict.image}`
                    );
                    info_pre.innerHTML = `Building : ${predict.building} , Area : ${predict.area}`

                  })
    
              })
      }
  }
}



document.querySelector('.close').addEventListener("click", function() {
document.querySelector('.bg-modal').style.display = "none";
});