const models = {
  "meta-llama/Llama-2-70b-chat-hf": {
    modelCard: "https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
    license: "https://bit.ly/llama2-license",
    maxSessionLength: 8192,
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "meta-llama/Llama-2-70b-hf": {
    modelCard: "https://huggingface.co/meta-llama/Llama-2-70b-hf",
    license: "https://bit.ly/llama2-license",
    maxSessionLength: 8192,
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "timdettmers/guanaco-65b": {
    modelCard: "https://huggingface.co/timdettmers/guanaco-65b",
    license: "https://huggingface.co/timdettmers/guanaco-65b",
    maxSessionLength: 2048,
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "enoch/llama-65b-hf": {
    modelCard: "https://github.com/facebookresearch/llama/blob/llama_v1/MODEL_CARD.md",
    license: "https://bit.ly/llama-license",
    maxSessionLength: 2048,
    sepToken: "###",
    stopToken: "###",
    extraStopSequences: ["</s>"],
  },
  "bigscience/bloom": {
    modelCard: "https://huggingface.co/bigscience/bloom",
    license: "https://bit.ly/bloom-license",
    maxSessionLength: 2048,
    sepToken: "\n\n",
    stopToken: "\n\n",
    extraStopSequences: null,
  },
  "bigscience/bloomz": {
    modelCard: "https://huggingface.co/bigscience/bloomz",
    license: "https://bit.ly/bloom-license",
    maxSessionLength: 2048,
    sepToken: "\n\n",
    stopToken: "</s>",
    extraStopSequences: ["\n\nHuman"],
  },
};
var curModel = "meta-llama/Llama-2-70b-chat-hf";

const generationParams = {
  do_sample: 1,
  temperature: 0.9,
  top_p: 0.6,
};

var ws = null;
var position = 0;
const initialSessionLength = 512;
var sessionLength = initialSessionLength;
var connFailureBefore = false;

var totalElapsed, nRequests, tokenCount;

const Regime = {
  CHATBOT: 1,
  FEW_SHOT: 2,
};
let curRegime = Regime.CHATBOT;
let stop = false;

function openSession() {
  let protocol = location.protocol == "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/api/v2/generate`);
  ws.onopen = () => {
    ws.send(JSON.stringify({type: "open_inference_session", model: curModel, max_length: sessionLength}));
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
      handleFailure(`Connection was closed`, true);
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
  tokenCount = 0;
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
    connFailureBefore = false;  // We've managed to connect after a possible failure

    const response = JSON.parse(event.data);
    if (!response.ok) {
      handleFailure(response.traceback);
      return;
    }

    if (lastMessageTime != null) {
      totalElapsed += performance.now() - lastMessageTime;
      tokenCount += response.token_count;
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
      if (tokenCount >= 1) {
        const speed = tokenCount / (totalElapsed / 1000);
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

function handleFailure(message, autoRetry = false) {
  resetSession();
  if (!isWaitingForInputs()) {
    // Show the error and the retry button only if a user is waiting for the generation results

    if (message === "Connection failed" && !connFailureBefore) {
      autoRetry = true;
      connFailureBefore = true;
    }
    if (/Session .+ expired/.test(message)) {
      autoRetry = true;
    }
    if (/Maximum length exceeded/.test(message) && sessionLength < models[curModel].maxSessionLength) {
      // We gradually increase sessionLength to save server resources. Default: 512 -> 2048 -> 8192 (if supported)
      sessionLength = Math.min(sessionLength * 4, models[curModel].maxSessionLength);
      autoRetry = true;
    }

    if (autoRetry) {
      retry();
    } else {
      $('.loading-animation').hide();
      if (/attention cache is full/.test(message)) {
        $('.error-message').hide();
        $('.out-of-capacity').show();
      } else {
        $('.out-of-capacity').hide();
        $('.error-message').text(message).show();
      }
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

  $('.family-selector label').click(function (e) {
    if (!isWaitingForInputs()) {
      alert("Can't switch the model while the AI is writing a response. Please refresh the page");
      e.preventDefault();
      return;
    }

    const radio = $(`#${$(this).attr("for")}`);
    if (radio.is(":checked")) {
      setTimeout(() => $('.human-replica textarea').focus(), 10);
      return;
    }

    const curFamily = radio.attr("value");
    $('.model-selector').hide();
    const firstLabel = $(`.model-selector[data-family=${curFamily}]`).show().children('label:first');
    firstLabel.click();
    firstLabel.trigger('click');
  });
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

    sessionLength = initialSessionLength;
    resetSession();
    appendTextArea();

    $('.model-name')
      .text($(this).text())
      .attr('href', models[curModel].modelCard);
    $('.license-link').attr('href', models[curModel].license);
    setTimeout(() => $('.human-replica textarea').focus(), 10);
  });
  $('.retry-link').click(e => {
    e.preventDefault();
    retry();
  });

  setInterval(animateLoading, 2000);
});
