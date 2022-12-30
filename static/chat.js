var sessionId = null;
var position = 0;

const sepToken = "\n\n";

function handleFailure(_request, _status, message) {
  const showError = !/Session .+ does not exist/.test(message);
  if (showError) {
    alert("Request failed. Retrying.\n" + message);
  }

  // Open a new inference session and regenerate the prefix
  sessionId = null;
  position = 0;
  sendReplica();
}

function sendReplica() {
  if (sessionId === null) {
    $.get('/api/v1/open_inference_session', null, null, "json")
      .done(data => {
        if (!data.ok) {
          handleFailure(null, null, data.traceback);
          return;
        }
        sessionId = data.session_id;
        sendReplica();
      })
      .fail(handleFailure);
    return;
  }

  const textarea = $('.human-replica:last textarea');
  if (textarea.length >= 1) {
    $('.human-replica:last').text(textarea.val());
    $('.dialogue').append($('<p class="ai-replica"><span class="text">AI:</span><span class="loading-animation"></span></p>'));
  }

  const replicaDivs = $('.human-replica, .ai-replica .text');
  var replicas = [];
  for (var i = position; i < replicaDivs.length; i++) {
    replicas.push($(replicaDivs[i]).text());
  }
  const inputs = replicas.join(sepToken);
  position = replicaDivs.length;

  receiveReplica(inputs);
}

const textareaHtml = '<p class="human-replica"><textarea class="form-control" id="exampleTextarea" rows="2">Human: </textarea></p>';

function receiveReplica(inputs) {
  const params = {
    max_new_tokens: 1,
    do_sample: 1,
    temperature: 0.75,
    top_p: 0.9,
    session_id: sessionId,
  };
  if (inputs !== null) {
    params.inputs = inputs;
  }

  $.post('/api/v1/generate', params, null, "json")
    .done(data => {
      if (!data.ok) {
        handleFailure(null, null, data.traceback);
        return;
      }

      const lastReplica = $('.ai-replica .text').last();
      const newText = lastReplica.text() + data.outputs;
      lastReplica.text(newText.replace(sepToken, ""));
      if (!newText.includes(sepToken)) {
        receiveReplica(null);
      } else {
        $('.loading-animation').remove();
        $('.dialogue').append($(textareaHtml));
        upgradeTextArea();
      }
    })
    .fail(handleFailure);
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
  sessionId = null;
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

  $('.clear-convo').click(e => {
    e.preventDefault();
    resetDialogue();
  });
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

  setInterval(animateLoading, 2000);
});
