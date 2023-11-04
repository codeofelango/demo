// 加入回车提交支持，shift+回车换行
var input = document.getElementById("chatinput");
input.addEventListener("keydown", function (event) {
    if (event.keyCode === 13 && !event.shiftKey) {
        event.preventDefault();
        document.getElementById("sendbutton").click();
    }
});




// Add your JavaScript here
document.getElementById("sendbutton").addEventListener("click", function () {
    let loading = document.getElementById('loading');
    loading.style.display = 'block';
    // Get the user's message from the input field
    var message = document.getElementById("chatinput").value;
    if (message.length < 1) {
        var response = document.createElement("div");
        response.innerHTML = "🤔<br>🤖<br>Message cannot be null";
        chatlog.appendChild(response);
        // Scroll the chatlog to the bottom
        chatlog.scrollTop = chatlog.scrollHeight;
        loading.style.display = 'none';
    } else {
        // Clear the input field
        document.getElementById("chatinput").value = "";
        // Send the message to the chatbot
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/chat/get?msg=" + message);
        xhr.send();
        xhr.onload = function () {
            // Append the chatbot's response to the chatlog
            var chatlog = document.getElementById("chatlog");
            var response = document.createElement("div");
            // response.innerHTML = "<div class='chat_container'><img src='' alt='Question:'><p>"+ message + "</p></div><br> \
            // <div class='chat_container chat_darker'><img src='' alt=':Answer' class='right'><p>" + marked.parse(xhr.responseText)+"</p>\
            // </div>";
            
            response.innerHTML = "User🤔: " + message + "<br>AI🤖: " + xhr.responseText;
        //     response.innerHTML = '<div class="message sent">User🤔:'+
        //     '<p>'+message+'</p>'+
        // '</div>'+
        // '<div class="message received">AI🤖:'+
        //     '<p>'+xhr.responseText+'</p>'+
        // '</div>'+

            // response.innerHTML = "🤔<br>" + message + "<br>🤖" + marked.parse(xhr.responseText);
            chatlog.appendChild(response);
            // Scroll the chatlog to the bottom
            chatlog.scrollTop = chatlog.scrollHeight;
            loading.style.display = 'none';
        }
    }
});


