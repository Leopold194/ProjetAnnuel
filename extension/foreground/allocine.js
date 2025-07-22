const ENDPOINT = "http://127.0.0.1:5000";

const container = document.querySelector(
  "#content-layout > section > div > div.card.entity-card.entity-card-list.cf.entity-card-player-ovw"
);

const url = document.querySelector(
  "#content-layout > section > div > div.card.entity-card.entity-card-list.cf.entity-card-player-ovw > figure > a > img"
).attributes["src"].value;

console.log(url);

chrome.storage.local.get(["selectedModel"], function (result) {
  console.log(result)
  const data = {
    url: url,
    model: result.selectedModel
  };
  console.log("Data to send:", data);

  predict(data);
});

function predict(data) {
  fetch(`${ENDPOINT}/api/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  })
    .then((response) => {
      console.log("Status code:", response.status);
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((json) => {
      console.log(json);
      var child = document.createElement("a");
      child.innerText = json["prediction"];
      child.className = "neural-cat";
      container.appendChild(child);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
