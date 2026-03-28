/**
 * WebSocket bridge for AgentTree Python orchestrator.
 *
 * Connects to the Python-side WebSocket server (localhost:9223).
 * Receives commands (create_tab, get_dom, click, etc.) and routes
 * them to the appropriate tab's content script or Chrome API.
 * Returns results back to the orchestrator.
 *
 * Uses Nanobrowser's existing BrowserContext and Page for tab management
 * and DOM serialization.
 */

import type BrowserContext from './browser/context';
import { createLogger } from './log';
import { injectBuildDomTreeScripts } from './browser/dom/service';
import { DEFAULT_AGENT_OPTIONS } from './agent/types';

const logger = createLogger('ws-bridge');

const RECONNECT_DELAY = 3000;
const KEEPALIVE_INTERVAL = 20_000;

interface WsCommand {
  id: string;
  type: string;
  tab_id?: string;
  params?: Record<string, unknown>;
}

interface WsResponse {
  id: string;
  type: 'result' | 'dom_state' | 'error' | 'captcha_detected';
  tab_id?: string;
  data: Record<string, unknown>;
}

export class AgentTreeBridge {
  private ws: WebSocket | null = null;
  private keepaliveTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private browserContext: BrowserContext;
  private port: number;
  private running = false;
  private managedTabs: Set<number> = new Set();

  constructor(browserContext: BrowserContext, port = 9223) {
    this.browserContext = browserContext;
    this.port = port;
  }

  start(): void {
    this.running = true;
    this.connect();
  }

  stop(): void {
    this.running = false;
    this.stopKeepalive();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private connect(): void {
    if (!this.running) return;

    try {
      this.ws = new WebSocket(`ws://localhost:${this.port}`);

      this.ws.onopen = () => {
        logger.info('AgentTree bridge connected to Python orchestrator');
        this.startKeepalive();
        // Announce our capabilities
        this.send({
          id: 'init',
          type: 'result',
          data: { status: 'connected', extension: 'agenttree-nanobrowser' },
        });
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data as string) as WsCommand;
          this.handleCommand(msg).catch(err => {
            logger.error('Error handling command:', err);
            this.send({
              id: msg.id,
              type: 'error',
              tab_id: msg.tab_id,
              data: { message: err instanceof Error ? err.message : String(err) },
            });
          });
        } catch (e) {
          logger.error('Failed to parse WebSocket message:', e);
        }
      };

      this.ws.onclose = () => {
        logger.info('AgentTree bridge disconnected');
        this.stopKeepalive();
        if (this.running) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error: Event) => {
        logger.error('WebSocket error:', error);
      };
    } catch (e) {
      logger.error('Failed to connect:', e);
      if (this.running) {
        this.scheduleReconnect();
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || !this.running) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, RECONNECT_DELAY);
  }

  private startKeepalive(): void {
    this.keepaliveTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, KEEPALIVE_INTERVAL);
  }

  private stopKeepalive(): void {
    if (this.keepaliveTimer) {
      clearInterval(this.keepaliveTimer);
      this.keepaliveTimer = null;
    }
  }

  private send(response: WsResponse): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(response));
    }
  }

  private async handleCommand(cmd: WsCommand): Promise<void> {
    const { id, type, tab_id: tabId, params } = cmd;

    switch (type) {
      case 'create_tab':
        await this.cmdCreateTab(id, params);
        break;
      case 'close_tab':
        await this.cmdCloseTab(id, tabId!);
        break;
      case 'navigate':
        await this.cmdNavigate(id, tabId!, params);
        break;
      case 'get_dom':
        await this.cmdGetDom(id, tabId!);
        break;
      case 'click':
        await this.cmdClick(id, tabId!, params);
        break;
      case 'type_text':
        await this.cmdTypeText(id, tabId!, params);
        break;
      case 'scroll':
        await this.cmdScroll(id, tabId!, params);
        break;
      case 'select':
        await this.cmdSelect(id, tabId!, params);
        break;
      case 'wait':
        await this.cmdWait(id, params);
        break;
      case 'extract':
        await this.cmdExtract(id, tabId!, params);
        break;
      case 'pong':
      case 'ping':
        // Keepalive — no response needed
        break;
      default:
        this.send({
          id,
          type: 'error',
          tab_id: tabId,
          data: { message: `Unknown command type: ${type}` },
        });
    }
  }

  // --- Command handlers ---

  private async cmdCreateTab(id: string, params?: Record<string, unknown>): Promise<void> {
    const url = (params?.url as string) || 'about:blank';
    const tab = await chrome.tabs.create({ url, active: false });
    const tabId = tab.id!;
    this.managedTabs.add(tabId);

    // Wait for tab to load and inject DOM scripts
    await new Promise<void>(resolve => {
      const listener = (changedTabId: number, info: chrome.tabs.TabChangeInfo) => {
        if (changedTabId === tabId && info.status === 'complete') {
          chrome.tabs.onUpdated.removeListener(listener);
          resolve();
        }
      };
      chrome.tabs.onUpdated.addListener(listener);
      setTimeout(() => {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }, 10000);
    });

    await injectBuildDomTreeScripts(tabId);

    this.send({
      id,
      type: 'result',
      tab_id: String(tabId),
      data: { tab_id: String(tabId), url },
    });
  }

  private async cmdCloseTab(id: string, tabId: string): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    if (!isNaN(numTabId)) {
      this.managedTabs.delete(numTabId);
      try {
        await chrome.tabs.remove(numTabId);
      } catch (e) {
        // Tab may already be closed
      }
    }
    this.send({ id, type: 'result', tab_id: tabId, data: { closed: true } });
  }

  private async cmdNavigate(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const url = params?.url as string;
    if (!url) {
      this.send({ id, type: 'error', tab_id: tabId, data: { message: 'url required' } });
      return;
    }

    await chrome.tabs.update(numTabId, { url });

    // Wait for navigation to complete
    await new Promise<void>(resolve => {
      const listener = (changedTabId: number, info: chrome.tabs.TabChangeInfo) => {
        if (changedTabId === numTabId && info.status === 'complete') {
          chrome.tabs.onUpdated.removeListener(listener);
          resolve();
        }
      };
      chrome.tabs.onUpdated.addListener(listener);
      setTimeout(() => {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }, 30000);
    });

    // Re-inject DOM scripts after navigation
    await injectBuildDomTreeScripts(numTabId);

    this.send({
      id,
      type: 'result',
      tab_id: tabId,
      data: { navigated: url },
    });
  }

  private async cmdGetDom(id: string, tabId: string): Promise<void> {
    const numTabId = parseInt(tabId, 10);

    try {
      // Use Nanobrowser's existing BrowserContext to get DOM state
      const page = await this.browserContext.switchTab(numTabId);
      const browserState = await this.browserContext.getState(false);

      // Serialize to the text format expected by AgentTree
      const elementsText = browserState.elementTree.clickableElementsToString(
        DEFAULT_AGENT_OPTIONS.includeAttributes,
      );

      const tab = await chrome.tabs.get(numTabId);
      const scrollInfo = browserState.scrollInfo;

      let domText = `Page: ${tab.url || ''}\n`;
      domText += `Title: ${tab.title || ''}\n\n`;
      domText += `[Interactive Elements]\n${elementsText}\n`;

      // Add scroll info
      if (scrollInfo) {
        const pagesBelow = ((scrollInfo.scrollHeight - scrollInfo.scrollTop - scrollInfo.viewportHeight) / scrollInfo.viewportHeight).toFixed(1);
        if (parseFloat(pagesBelow) > 0.1) {
          domText += `\n|SCROLL| (${pagesBelow} pages below)`;
        }
      }

      this.send({
        id,
        type: 'dom_state',
        tab_id: tabId,
        data: {
          dom: domText,
          url: tab.url || '',
          title: tab.title || '',
          captcha_detected: false,
        },
      });
    } catch (err) {
      // Fallback: simple DOM extraction via executeScript
      try {
        const [result] = await chrome.scripting.executeScript({
          target: { tabId: numTabId },
          func: () => {
            const interactiveEls = document.querySelectorAll(
              'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]',
            );
            const elements: string[] = [];
            let counter = 0;
            interactiveEls.forEach(el => {
              const rect = el.getBoundingClientRect();
              if (rect.width > 0 && rect.height > 0) {
                counter++;
                const tag = el.tagName.toLowerCase();
                const attrs: string[] = [];
                for (const attr of ['type', 'placeholder', 'href', 'value', 'name', 'role']) {
                  const val = el.getAttribute(attr);
                  if (val) attrs.push(`${attr}="${val.slice(0, 80)}"`);
                }
                const text = (el.textContent || '').trim().slice(0, 100);
                const attrStr = attrs.length > 0 ? ' ' + attrs.join(' ') : '';
                elements.push(`[e${counter}] <${tag}${attrStr}>${text}</${tag}>`);
              }
            });

            const bodyText = (document.body?.innerText || '').slice(0, 8000);
            return {
              elements: elements.join('\n'),
              text: bodyText,
              url: window.location.href,
              title: document.title,
            };
          },
        });

        const data = result.result as { elements: string; text: string; url: string; title: string };
        const domText = `Page: ${data.url}\nTitle: ${data.title}\n\n[Interactive Elements]\n${data.elements}\n\n[Page Content]\n${data.text}`;

        this.send({
          id,
          type: 'dom_state',
          tab_id: tabId,
          data: {
            dom: domText,
            url: data.url,
            title: data.title,
            captcha_detected: false,
          },
        });
      } catch (fallbackErr) {
        this.send({
          id,
          type: 'error',
          tab_id: tabId,
          data: { message: `DOM extraction failed: ${fallbackErr}` },
        });
      }
    }
  }

  private async cmdClick(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const ref = params?.ref as string;

    if (!ref) {
      this.send({ id, type: 'error', tab_id: tabId, data: { message: 'ref required' } });
      return;
    }

    // Extract the numeric index from ref (e.g., "e5" -> 5)
    const indexMatch = ref.match(/\d+/);
    if (!indexMatch) {
      this.send({ id, type: 'error', tab_id: tabId, data: { message: `Invalid ref: ${ref}` } });
      return;
    }
    const index = parseInt(indexMatch[0], 10);

    try {
      const page = await this.browserContext.switchTab(numTabId);
      await page.click(index);
      this.send({ id, type: 'result', tab_id: tabId, data: { clicked: ref, success: true } });
    } catch (err) {
      // Fallback: use executeScript
      try {
        await chrome.scripting.executeScript({
          target: { tabId: numTabId },
          func: (refStr: string) => {
            const idx = parseInt(refStr.replace(/\D/g, ''), 10);
            const interactiveEls = document.querySelectorAll(
              'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]',
            );
            const visibleEls = Array.from(interactiveEls).filter(el => {
              const rect = el.getBoundingClientRect();
              return rect.width > 0 && rect.height > 0;
            });
            if (idx > 0 && idx <= visibleEls.length) {
              (visibleEls[idx - 1] as HTMLElement).click();
            }
          },
          args: [ref],
        });
        this.send({ id, type: 'result', tab_id: tabId, data: { clicked: ref, success: true } });
      } catch (e) {
        this.send({
          id,
          type: 'error',
          tab_id: tabId,
          data: { message: `Click failed: ${e}` },
        });
      }
    }
  }

  private async cmdTypeText(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const ref = params?.ref as string;
    const text = params?.text as string;
    const clearFirst = params?.clear_first !== false;

    if (!ref || text === undefined) {
      this.send({
        id,
        type: 'error',
        tab_id: tabId,
        data: { message: 'ref and text required' },
      });
      return;
    }

    try {
      // Use executeScript for reliable text input
      await chrome.scripting.executeScript({
        target: { tabId: numTabId },
        func: (refStr: string, inputText: string, clear: boolean) => {
          const idx = parseInt(refStr.replace(/\D/g, ''), 10);
          const interactiveEls = document.querySelectorAll(
            'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]',
          );
          const visibleEls = Array.from(interactiveEls).filter(el => {
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          });
          if (idx > 0 && idx <= visibleEls.length) {
            const el = visibleEls[idx - 1] as HTMLInputElement;
            el.focus();
            if (clear) el.value = '';
            el.value += inputText;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
        },
        args: [ref, text, clearFirst],
      });
      this.send({ id, type: 'result', tab_id: tabId, data: { typed: text, success: true } });
    } catch (e) {
      this.send({
        id,
        type: 'error',
        tab_id: tabId,
        data: { message: `Type failed: ${e}` },
      });
    }
  }

  private async cmdScroll(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const direction = params?.direction as string || 'down';
    const amount = (params?.amount as number) || 500;

    try {
      await chrome.scripting.executeScript({
        target: { tabId: numTabId },
        func: (dir: string, amt: number) => {
          window.scrollBy(0, dir === 'up' ? -amt : amt);
        },
        args: [direction, amount],
      });
      this.send({ id, type: 'result', tab_id: tabId, data: { scrolled: direction, success: true } });
    } catch (e) {
      this.send({
        id,
        type: 'error',
        tab_id: tabId,
        data: { message: `Scroll failed: ${e}` },
      });
    }
  }

  private async cmdSelect(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const ref = params?.ref as string;
    const value = params?.value as string;

    if (!ref || !value) {
      this.send({ id, type: 'error', tab_id: tabId, data: { message: 'ref and value required' } });
      return;
    }

    try {
      await chrome.scripting.executeScript({
        target: { tabId: numTabId },
        func: (refStr: string, val: string) => {
          const idx = parseInt(refStr.replace(/\D/g, ''), 10);
          const interactiveEls = document.querySelectorAll(
            'a, button, input, select, textarea, [role="button"], [onclick], [tabindex]',
          );
          const visibleEls = Array.from(interactiveEls).filter(el => {
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          });
          if (idx > 0 && idx <= visibleEls.length) {
            const el = visibleEls[idx - 1] as HTMLSelectElement;
            el.value = val;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
        },
        args: [ref, value],
      });
      this.send({ id, type: 'result', tab_id: tabId, data: { selected: value, success: true } });
    } catch (e) {
      this.send({
        id,
        type: 'error',
        tab_id: tabId,
        data: { message: `Select failed: ${e}` },
      });
    }
  }

  private async cmdWait(id: string, params?: Record<string, unknown>): Promise<void> {
    const seconds = Math.min((params?.seconds as number) || 2, 30);
    await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    this.send({ id, type: 'result', data: { waited: seconds, success: true } });
  }

  private async cmdExtract(id: string, tabId: string, params?: Record<string, unknown>): Promise<void> {
    const numTabId = parseInt(tabId, 10);
    const goal = params?.goal as string || 'Extract all relevant data from the page';

    try {
      const [result] = await chrome.scripting.executeScript({
        target: { tabId: numTabId },
        func: () => {
          return {
            url: window.location.href,
            title: document.title,
            text: (document.body?.innerText || '').slice(0, 16000),
          };
        },
      });
      const data = result.result as { url: string; title: string; text: string };
      this.send({
        id,
        type: 'result',
        tab_id: tabId,
        data: { extracted: data, goal, success: true },
      });
    } catch (e) {
      this.send({
        id,
        type: 'error',
        tab_id: tabId,
        data: { message: `Extract failed: ${e}` },
      });
    }
  }

  // Called by CAPTCHA detection content script
  reportCaptcha(tabId: number, patterns: string[]): void {
    this.send({
      id: `captcha_${Date.now()}`,
      type: 'captcha_detected',
      tab_id: String(tabId),
      data: { patterns },
    });
  }
}
