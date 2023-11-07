const videoElement = document.getElementById('camera-feed');
const captureButton = document.getElementById('capture-button');
const recordButton = document.getElementById('record-button');
const fullscreenButton = document.getElementById('fullscreen-button');
const imageGallery = document.getElementById('image-gallery');
let mediaStream;
let mediaRecorder;
let chunks = [];
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        mediaStream = stream;
        videoElement.srcObject = stream;
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.onstart = () => {
            chunks = [];
        };
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };
        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { 'type': 'video/mp4' });
            const videoURL = URL.createObjectURL(blob);

            const videoElement = document.createElement('video');
            videoElement.controls = true;
            videoElement.src = videoURL;

            const deleteButton = createDeleteButton(videoElement);

            const mediaContainer = document.createElement('div');
            mediaContainer.appendChild(videoElement);
            mediaContainer.appendChild(deleteButton);
            imageGallery.appendChild(mediaContainer);
        };
    })
    .catch((error) => {
        console.error('Không thể truy cập máy ảnh: ' + error);
    });
function deleteMedia(elementToDelete) {
    elementToDelete.parentElement.remove();
}
function createDeleteButton(videoElement) {
    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'Xóa';
    deleteButton.addEventListener('click', () => {
        deleteMedia(videoElement);
    });
    return deleteButton;
}
captureButton.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const imgData = canvas.toDataURL('image/png');
    const imgElement = document.createElement('img');
    imgElement.src = imgData;
    const deleteButton = createDeleteButton(imgElement);
    const mediaContainer = document.createElement('div');
    mediaContainer.appendChild(imgElement);
    mediaContainer.appendChild(deleteButton);
    imageGallery.appendChild(mediaContainer);
});
recordButton.addEventListener('click', () => {
    if (mediaRecorder.state === 'inactive') {
        mediaRecorder.start();
        recordButton.textContent = 'Dừng Quay Video';
    } else if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordButton.textContent = 'Bắt Đầu Quay Video';
    }
});
fullscreenButton.addEventListener('click', () => {
    if (videoElement.requestFullscreen);
        videoElement.requestFullscreen();
})
   
