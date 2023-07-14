const models = {
  "timdettmers/guanaco-65b": {
    modelCard: "https://huggingface.co/timdettmers/guanaco-65b",
    license: "https://huggingface.co/timdettmers/guanaco-65b",
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "enoch/llama-65b-hf": {
    modelCard: "https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md",
    license: "https://bit.ly/llama-license",
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "bigscience/bloom": {
    modelCard: "https://huggingface.co/bigscience/bloom",
    license: "https://bit.ly/bloom-license",
    sepToken: "\n\n",
    stopToken: "\n\n",
    extraStopSequences: null,
  },
  "bigscience/bloomz": {
    modelCard: "https://huggingface.co/bigscience/bloomz",
    license: "https://bit.ly/bloom-license",
    sepToken: "\n\n",
    stopToken: "</s>",
    extraStopSequences: ["\n\nHuman"],
  },
};
var curModel = "timdettmers/guanaco-65b";

const generationParams = {
  do_sample: 1,
  temperature: 0.7,
  top_k: 40,
};

var ws = null;
var position = 0;
var sessionMaxLength = 1024;

var totalElapsed, nRequests;

const Regime = {
  CHATBOT: 1,
  FEW_SHOT: 2,
};
let curRegime = Regime.CHATBOT;
let stop = false;

function openSession() {
  ws = new WebSocket(`ws://${location.host}/api/v2/generate`);
  ws.onopen = () => {
    ws.send(JSON.stringify({type: "open_inference_session", model: curModel, max_length: sessionMaxLength}));
    ws.onmessage = event => {
      const response = JSON.parse(event.data);
      if (!response.ok) {
        handleFailure(response.traceback);
        return;
      }

      sendReplica();
    };
  };

  ws.onerror = _event => handleFailure(`Connection failed`);
  ws.onclose = _event => {
    if ($(".error-box").is(":hidden")) {
      handleFailure(`Connection was closed`);
    }
  };
}

function resetSession() {
  if (ws !== null && ws.readyState <= 1) {  // If readyState is "connecting" or "opened"
    ws.close();
  }
  ws = null;
  position = 0;
}

function isWaitingForInputs() {
  return $('.human-replica textarea').length >= 1;
}

function sendReplica() {
  if (isWaitingForInputs()) {
    const aiPrompt = (curRegime === Regime.CHATBOT) ? 'Assistant:' : '';
    $('.human-replica:last').text($('.human-replica:last textarea').val());
    $('.dialogue').append($(
      '<p class="ai-replica">' +
        `<span class="text">${aiPrompt}</span>` +
        '<span class="loading-animation"></span>' +
        '<span class="speed" style="display: none;"></span>' +
        '<span class="generation-controls"><a class="stop-generation" href=#>stop generation</a></span>' +
        '<span class="suggest-join" style="display: none;">' +
          '<b>Too slow?</b> ' +
          '<a target="_blank" href="https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity">Connect your GPU</a> ' +
          'and increase Petals capacity!' +
        '</span>' +
      '</p>'));
    animateLoading();
    $('.stop-generation').click(e => {
      e.preventDefault();
      console.log("Stop generation");
      stop = true;
    });
  } else {
    $('.loading-animation').show();
  }

  if (ws === null) {
    openSession();
    return;
  }

  const replicaDivs = $('.human-replica, .ai-replica .text');
  var replicas = [];
  for (var i = position; i < replicaDivs.length; i++) {
    const el = $(replicaDivs[i]);
    var phrase = el.text();
    if (el.is(".human-replica")) {
      phrase += models[curModel].sepToken;
    } else
    if (i < replicaDivs.length - 1) {
      phrase += models[curModel].stopToken;
    }
    replicas.push(phrase);
  }
  const inputs = replicas.join("");
  position = replicaDivs.length;

  totalElapsed = 0;
  nRequests = 0;
  receiveReplica(inputs);
}

function receiveReplica(inputs) {
  ws.send(JSON.stringify({
    type: "generate",
    inputs: inputs,
    max_new_tokens: 1,
    stop_sequence: models[curModel].stopToken,
    extra_stop_sequences: models[curModel].extraStopSequences,
    ...generationParams
  }));

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
    var newText = lastReplica.text() + response.outputs;
    newText = newText.replace(models[curModel].stopToken, "");
    if (models[curModel].extraStopSequences !== null) {
      for (const seq of models[curModel].extraStopSequences) {
        newText = newText.replace(seq, "");
      }
    }
    lastReplica.text(newText);

    if (!response.stop && !stop) {
      if (nRequests >= 1) {
        const speed = nRequests / (totalElapsed / 1000);
        $('.speed')
          .text(`Speed: ${speed.toFixed(1)} tokens/sec`)
          .show();
        if (speed < 0.5) {
          $('.suggest-join').show();
        }
      }
    } else {
      $('.loading-animation, .speed, .suggest-join, .generation-controls').remove();
      resetSession();
      appendTextArea();
      stop = false;
    }
  };
}

function handleFailure(message) {
  resetSession();
  if (!isWaitingForInputs()) {
    // Show the error and the retry button only if a user is waiting for the generation results
    var autoRetry = false;
    if (/Session .+ expired/.test(message)) {
      autoRetry = true;
    }
    const largerMaxLength = 2048;
    if (/Maximum length exceeded/.test(message) && sessionMaxLength < largerMaxLength) {
      sessionMaxLength = largerMaxLength;  // We gradually increase sessionMaxLength to save server resources
      autoRetry = true;
    }

    if (autoRetry) {
      retry();
    } else {
      $('.loading-animation').hide();
      $('.error-message').text(message);
      $('.error-box').show();
    }
  }
}

function retry() {
  $('.error-box').hide();
  sendReplica();
}

function appendTextArea() {
  const humanPrompt = (curRegime === Regime.CHATBOT) ? "Human: " : "";
  $('.dialogue').append($(
    `<p class="human-replica"><textarea class="form-control" id="exampleTextarea" rows="2">${humanPrompt}</textarea></p>`
  ));
  upgradeTextArea();
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

const animFrames = ["⌛", "🧠"];
var curFrame = 0;

function animateLoading() {
  $('.loading-animation').html(' &nbsp;' + animFrames[curFrame]);
  curFrame = (curFrame + 1) % animFrames.length;
}

$(() => {
  upgradeTextArea();

  $('.model-selector label').click(function (e) {
    if (!isWaitingForInputs()) {
      alert("Can't switch the model while the AI is writing a response. Please refresh the page");
      e.preventDefault();
      return;
    }

    curModel = $(`#${$(this).attr("for")}`).attr("value");
    if (curRegime === Regime.CHATBOT) {
      $('.dialogue p').slice(2).remove();
    } else {
      $('.dialogue').empty();
    }
    resetSession();
    appendTextArea();

    $('.model-name')
      .text($(this).text())
      .attr('href', models[curModel].modelCard);
    $('.license-link').attr('href', models[curModel].license);
    setTimeout(() => $('.human-replica textarea').focus(), 10);
  });
  $('.regime-selector label').click(function (e) {
    if (!isWaitingForInputs()) {
      alert("Can't switch the regime while the AI is writing a response. Please refresh the page");
      e.preventDefault();
      return;
    }

    $('.dialogue').empty();
    if ($(this).attr("for") === "regime-chatbot") {
      location.reload();
      return;
    }

    curRegime = Regime.FEW_SHOT;
    resetSession();
    appendTextArea();

    const textarea = $('.human-replica textarea');
    textarea.val(
      'Input: A cat sat on a mat.\n\n' +
      'Output: Un gato se sentó en una estera.\n\n' +
      'Input: A brown fox jumps over the lazy dog.\n\n' +
      'Output: Un zorro marrón salta sobre el perro perezoso.\n\n' +
      'Input: Who is the president of the United States?'
    );
    textarea[0].style.height = textarea[0].scrollHeight + "px";
    setTimeout(() => textarea.focus(), 10);
  });
  $('.retry-link').click(e => {
    e.preventDefault();
    retry();
  });

  setInterval(animateLoading, 2000);
});
