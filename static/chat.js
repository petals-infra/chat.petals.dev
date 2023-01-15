var ws = null;
var position = 0;

var totalElapsed, nRequests;

const sepToken = "\n\n";

function sendReplica() {
  if (ws === null) {
    ws = new WebSocket(`ws://${location.host}/api/v2/generate`);
    ws.onopen = () => {
      ws.send(JSON.stringify({type: "open_inference_session", max_length: 1024}));
      ws.onmessage = event => {
        const response = JSON.parse(event.data);
        if (!response.ok) {
          handleFailure(response.traceback);
          return;
        }

        sendReplica();
      };
    };
    ws.onclose = event => handleFailure(`Connection closed (reason="${event.reason}", code=${event.code})`);
    return;
  }

  const textarea = $('.human-replica:last textarea');
  if (textarea.length >= 1) {
    $('.human-replica:last').text(textarea.val());
    $('.dialogue').append($(
      '<p class="ai-replica">' +
        '<span class="text">AI:</span><span class="loading-animation"></span>' +
        '<span class="speed" style="display: none;">Average speed: <span class="value"></span> sec/token</span>' +
        '<span class="suggest-join" style="display: none;">' +
          'This speed is slower than expected due to a high load. You can increase Petals capacity by ' +
          '<a target="_blank" href="https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity">connecting your GPU</a>.' +
        '</span>' +
      '</p>'));
  } else {
    $('.loading-animation').show();
  }

  const replicaDivs = $('.human-replica, .ai-replica .text');
  var replicas = [];
  for (var i = position; i < replicaDivs.length; i++) {
    replicas.push($(replicaDivs[i]).text());
  }
  const inputs = replicas.join(sepToken);
  position = replicaDivs.length;

  totalElapsed = 0;
  nRequests = 0;
  receiveReplica(inputs);
}

const textareaHtml = '<p class="human-replica"><textarea class="form-control" id="exampleTextarea" rows="2">Human: </textarea></p>';

function receiveReplica(inputs) {
  const request = {
    type: "generate",
    max_new_tokens: 1,
    do_sample: 1,
    temperature: 0.75,
    top_p: 0.9,
    session_id: ws,
    stop_sequence: sepToken,
  };
  if (inputs !== null) {
    request.inputs = inputs;
  }

  ws.send(JSON.stringify(request));

  var lastMessageTime = null;
  ws.onmessage = event => {
    const response = JSON.parse(event.data);
    if (!response.ok) {
      handleFailure(response.traceback);
      return;
    }

    if (lastMessageTime != null) {
      totalElapsed += performance.now() - lastMessageTime;
      nRequests++;
    }
    lastMessageTime = performance.now();

    const lastReplica = $('.ai-replica .text').last();
    const newText = lastReplica.text() + response.outputs;
    lastReplica.text(newText.replace(sepToken, ""));
    if (!response.stop) {
      if (nRequests >= 1) {
        const stepsPerSecond = totalElapsed / nRequests / 1000;
        $('.speed .value').text(stepsPerSecond.toFixed(1));
        $('.speed').show();
        if (stepsPerSecond >= 3) {
          $('.suggest-join').show();
        }
      }
    } else {
      $('.loading-animation, .speed, .suggest-join').remove();
      $('.dialogue').append($(textareaHtml));
      upgradeTextArea();
    }
  };
}

function handleFailure(message) {
  const showError = !/Session .+ expired/.test(message);
  if (showError) {
    $('.loading-animation').hide();
    $('.error-message').text(message);
    $('.error-box').show();
  } else {
    retry();
  }
}

function retry() {
  $('.error-box').hide();

  // Open a new inference session and regenerate the prefix
  if (ws != null) {
    ws.close();
  }
  ws = null;
  position = 0;
  sendReplica();
}

function upgradeTextArea() {
  const textarea = $('.human-replica textarea');
  autosize(textarea);
  textarea[0].selectionStart = textarea[0].value.length;
  textarea.focus();

  textarea.on('keypress', e => {
    if (e.which == 13 && !e.shiftKey) {
      e.preventDefault();
      sendReplica();
    }
  });
}

function resetDialogue() {
  if ($('.human-replica textarea').length == 0) {
    alert("Can't reset the dialogue while the AI is writing a response.");
    return false;
  }
  if (!confirm("This will reset the dialogue. Are you sure?")) {
    return false;
  }

  $('.dialogue').html(textareaHtml);
  upgradeTextArea();

  if (ws != null) {
    ws.close();
  }
  ws = null;
  position = 0;
  return true;
}

const animFrames = ["⌛", "🧠"];
var curFrame = 0;

function animateLoading() {
  $('.loading-animation').html(' &nbsp;' + animFrames[curFrame]);
  curFrame = (curFrame + 1) % animFrames.length;
}

$(() => {
  upgradeTextArea();

  $('.show-few-shot').click(e => {
    e.preventDefault();
    if (resetDialogue()) {
      const textarea = $('.human-replica textarea');
      textarea.val(
        'Human: A cat sat on a mat.\n\n' +
        'AI: Un gato se sentó en una estera.\n\n' +
        'Human: A brown fox jumps over the lazy dog.\n\n' +
        'AI: Un zorro marrón salta sobre el perro perezoso.\n\n' +
        'Human: Who is the president of the United States?'
      );
      textarea[0].style.height = textarea[0].scrollHeight + "px";
    }
  });
  $('.retry-link').click(e => {
    e.preventDefault();
    retry();
  });

  setInterval(animateLoading, 2000);
});
