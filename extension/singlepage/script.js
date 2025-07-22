const ENDPOINT = "http://localhost:5000";

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
    dropzone.innerHTML = ''; // vide la zone
    const img = document.createElement('img');
    img.src = url;
    img.style.maxWidth = '100%';
    img.style.maxHeight = '300px';
    dropzone.appendChild(img);
  }

  function predict(url) {
    chrome.storage.local.get(["selectedModel"], function (result) {
        console.log(result)
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
        showImagePreview(imageUrl);
        predict(imageUrl);
      } else {
        console.error("Échec de l'upload", data);
      }
    })
    .catch((err) => console.error("Erreur upload :", err));
}


document.getElementById("result-container").style.display = "none";