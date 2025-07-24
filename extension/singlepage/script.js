const ENDPOINT =
  "https://cineaiwn3fwc3j-cineai-api.functions.fnc.fr-par.scw.cloud";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("hover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("hover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("hover");
  handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

function showImagePreview(url) {
  dropzone.innerHTML = ""; // vide la zone
  const img = document.createElement("img");
  img.src = url;
  img.style.maxWidth = "100%";
  img.style.maxHeight = "300px";
  dropzone.appendChild(img);
}

function predict(url) {
  chrome.storage.local.get(["selectedModel"], function (result) {
    console.log(result);
    const data = {
      url: url,
      model: result.selectedModel
    };
    console.log("Data to send:", data);

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
        const resultContainer = document.getElementById("result");
        resultContainer.innerText = json["prediction"];
        document.getElementById("result-container").style.display = "block";
        document.getElementById("result-container").classList.remove("fade-in");
        void document.getElementById("result-container").offsetWidth;
        document.getElementById("result-container").classList.add("fade-in");
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });
}

function handleFile(file) {
  if (!file) return;

  const allowedTypes = [
    "image/avif",
    "image/bmp",
    "image/dds",
    "image/gif",
    "image/icns",
    "image/x-icon",
    "image/jpeg",
    "image/jp2",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/x-portable-pixmap",
    "image/x-tga",
    "image/vnd.ms-photo"
  ];

  if (
    !allowedTypes.includes(file.type) &&
    !file.name.match(
      /\.(blp|dib|eps|im|mpo|msp|pcx|pfm|ppm|qoi|sgi|spider|xbm)$/i
    )
  ) {
    alert("Format non supporté");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", file.name);

  NProgress.start();

  fetch(`${ENDPOINT}/api/images/upload`, {
    method: "POST",
    body: formData
  })
    .then(async (res) => {
      const contentType = res.headers.get("content-type");
      const data =
        contentType && contentType.includes("application/json")
          ? await res.json()
          : await res.text();

      if (res.ok && data.uuid) {
        const imageUrl = `${ENDPOINT}/api/document/retrieve/${data.uuid}`;
        document.getElementById("result-container").style.display = "none";
        showImagePreview(imageUrl);
        predict(imageUrl);
      } else {
        console.error("Échec de l'upload", data);
      }
      NProgress.done();
    })
    .catch((err) => {
      NProgress.done(), console.error("Erreur upload :", err);
    });
}

document.getElementById("result-container").style.display = "none";

// UPLOAD MODEL

async function upload(file, name) {
  const url = `${ENDPOINT}/api/model/upload`;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", name);
  formData.append("model_type", "auto");

  const response = await fetch(url, {
    method: "POST",
    body: formData
  });

  let data;
  try {
    data = await response.json();
  } catch {
    data = await response.text();
  }

  return { status: response.status, data };
}

var keys = "";
document.addEventListener("keydown", (e) => {
  keys += e.key;
  if (keys.endsWith("iknowwhatimdoing")) {
    keys = "";
    const uploadBtn = document.getElementById("upload-button-container");
    uploadBtn.classList.remove("hidden");
  }
  if (keys.length > 100) {
    keys = keys.slice(1);
  }
  console.log(keys);
});

const uploadBtn = document.getElementById("upload-model-button");

uploadBtn.addEventListener("click", () => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "application/json";
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const name = prompt("Enter model name:");
    if (!name) return;

    NProgress.start();

    const result = await upload(file, name);
    if (result.status >= 200 && result.status < 300) {
      alert(`Upload successful: ${JSON.stringify(result.data)}`);
    } else {
      alert(`Upload failed (${result.status}): ${JSON.stringify(result.data)}`);
    }
    NProgress.done();
  };
  input.click();
});
