"""No real sockets: urllib handler double for subscription redirect policy."""
import email.message
import io
import urllib.error
import urllib.request
import pytest

from lib.hal import gateway, gateway_egress as egress
from lib.hal.cascade_target import BASE_URL, PRODUCT, PROVIDER, SELECTOR, validate


def binding():
    return validate({'provider': PROVIDER, 'product': PRODUCT, 'model': 'qwen3.8-max',
                     'base_url': BASE_URL, 'credential': SELECTOR, 'auth': 'bearer',
                     'billing': 'subscription', 'transport': 'openai-chat'})


@pytest.mark.parametrize('entry', ['same', 'cross'])
def test_bound_egress_refuses_redirect_without_injected_opener(monkeypatch, entry):
    calls, guarded = [], []
    class FakeHTTPS(urllib.request.HTTPSHandler):
        def __init__(self, clock):
            super().__init__()
        def https_open(self, request):
            calls.append((request.full_url, request.get_header('Authorization')))
            headers = email.message.Message()
            headers['Location'] = 'https://api.openai.com/v1/chat/completions'
            answer = urllib.response.addinfourl(io.BytesIO(b''), headers,
                                                request.full_url, code=302)
            answer.msg = 'redirect'
            return answer
    monkeypatch.setattr(gateway, '_TimedHTTPSHandler', FakeHTTPS)
    monkeypatch.setattr(egress, 'require_openai_destination', guarded.append)
    target = binding()
    assert target is not None
    if entry == 'same':
        response = egress.forward_same_wire(PROVIDER, '/v1/chat/completions',
            b'{"model":"qwen3.8-max"}', binding=target,
            key_resolver=lambda _: 'synthetic-plan-key')
    else:
        response = egress.cross_wire_forward({'model': target.model, 'max_tokens': 8,
            'messages': [{'role': 'user', 'content': 'hello'}]}, provider=PROVIDER,
            want_stream=False, binding=target,
            key_resolver=lambda _: 'synthetic-plan-key')
    assert response.status == 302
    assert calls == [(BASE_URL + '/chat/completions', 'Bearer synthetic-plan-key')]
    assert guarded == [BASE_URL + '/chat/completions']




def test_bound_opener_does_not_follow_redirect(monkeypatch):
    calls = []
    class FakeHTTPS(urllib.request.HTTPSHandler):
        def __init__(self, clock):
            super().__init__()
        def https_open(self, request):
            calls.append(request.full_url)
            headers = email.message.Message()
            headers['Location'] = 'https://paid.example/v1/chat/completions'
            response = urllib.response.addinfourl(io.BytesIO(b''), headers,
                                                  request.full_url, code=302)
            response.msg = 'redirect'
            return response
    monkeypatch.setattr(gateway, '_TimedHTTPSHandler', FakeHTTPS)
    response = gateway.build_default_opener(subscription_bound=True)(
        'POST', 'https://plan.example/v1/chat/completions',
        [('Authorization', 'Bearer synthetic-plan-key')], b'{}')
    assert response.status == 302
    assert calls == ['https://plan.example/v1/chat/completions']


def test_ordinary_opener_keeps_redirect_handler(monkeypatch):
    handlers = []
    def build(*items):
        handlers.extend(items)
        return type('FakeOpener', (), {'open': lambda self, *a, **kw: io.BytesIO(b'')})()
    monkeypatch.setattr(urllib.request, 'build_opener', build)
    gateway.build_default_opener()('GET', 'https://fake.example/v1/models', [], None)
    assert not any(isinstance(item, gateway._NoRedirect) for item in handlers)
