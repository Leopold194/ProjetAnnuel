const ENDPOINT = "http://127.0.0.1:5000"


function list_model() {
    return fetch(`${ENDPOINT}/api/models/list`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then((response) => {
        console.log("Status code:", response.status);
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
}

list_model().then(models => {
    console.log(models);
    
    const dropdown = document.getElementById("model-selector");
    
    models["models"].forEach(model => {
        const option = document.createElement("option");
        option.value = model['name'];
        option.textContent = model['name'];
        dropdown.appendChild(option);
    });

    chrome.storage.local.get("selectedModel", function(data) {
        console.log(data)   
        // select it in the dropdown
        const dropdown = document.getElementById("model-selector");
        if (data.selectedModel) {
            dropdown.value = data.selectedModel;
            console.log("Selected model from storage:", data.selectedModel);
        }
        else {
            console.log("No model selected, using default.");
        }
    });
    
}).catch(error => {
    console.error("Error fetching models:", error);
});


document.getElementById("model-selector").addEventListener("change", function() {
    const selectedModel = this.value;
    console.log("Selected model:", selectedModel);

    // set the selected model in the local storage
    chrome.storage.local.set({ selectedModel: selectedModel }, function() {
        console.log("Selected model saved:", selectedModel);
    });

    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        chrome.tabs.reload(tabs[0].id);
      });
      
});

document.getElementById('open-page-btn').addEventListener('click', () => {
    chrome.tabs.create({ url: chrome.runtime.getURL('singlepage/index.html') });
  });
  

