// 検索（ブラウザ内で完結。assets/index.js の SEARCH_INDEX を使う）と、表の並べ替え
(function () {
  const norm = (s) => s.normalize("NFKC").toLowerCase();
  const KIND = { h: "馬", j: "騎手", t: "調教師", d: "開催日", r: "レース" };
  const INDEX = (window.SEARCH_INDEX || []).map((e) => ({ kind: e[0], name: e[1], url: e[2], note: e[3] || "", key: norm(e[1]) }));

  function search(q, limit) {
    q = norm(q.trim());
    if (!q) return [];
    const hit = [];
    for (const e of INDEX) {
      const i = e.key.indexOf(q);
      if (i < 0) continue;
      const rank = (i === 0 ? 0 : 1) + (e.kind === "h" ? 0 : e.kind === "d" ? 0.2 : 0.1);
      hit.push([rank, e]);
    }
    hit.sort((a, b) => a[0] - b[0] || a[1].name.length - b[1].name.length);
    return hit.slice(0, limit).map((x) => x[1]);
  }

  const root = document.documentElement.dataset.root || "";
  const li = (e) => `<a href="${root}${e.url}"><span>${e.name}</span><small>${KIND[e.kind] || ""}${e.note ? "・" + e.note : ""}</small></a>`;

  // ヘッダーの検索ボックス（候補表示）
  const form = document.querySelector("form.search");
  if (form) {
    const input = form.querySelector("input");
    const box = document.createElement("div");
    box.className = "suggest";
    form.appendChild(box);
    let sel = -1;
    input.addEventListener("input", () => {
      const r = search(input.value, 12);
      box.innerHTML = r.map(li).join("");
      box.classList.toggle("on", r.length > 0);
      sel = -1;
    });
    input.addEventListener("keydown", (ev) => {
      const items = [...box.querySelectorAll("a")];
      if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
        ev.preventDefault();
        if (!items.length) return;
        sel = (sel + (ev.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        items.forEach((a, i) => a.classList.toggle("sel", i === sel));
      } else if (ev.key === "Enter" && sel >= 0) {
        ev.preventDefault();
        location.href = items[sel].href;
      }
    });
    document.addEventListener("click", (ev) => { if (!form.contains(ev.target)) box.classList.remove("on"); });
  }

  // 検索ページ
  const out = document.getElementById("search-results");
  if (out) {
    const q = new URLSearchParams(location.search).get("q") || "";
    const input = document.getElementById("q");
    input.value = q;
    const show = () => {
      const r = search(input.value, 300);
      out.innerHTML = r.length
        ? `<p class="muted">${r.length >= 300 ? "上位300件" : r.length + "件"}</p><div class="table-wrap"><table><tbody>${r.map((e) => `<tr><td><a href="${root}${e.url}">${e.name}</a></td><td class="muted">${KIND[e.kind] || ""}</td><td class="muted">${e.note}</td></tr>`).join("")}</tbody></table></div>`
        : input.value ? "<p class='muted'>見つかりませんでした。</p>" : "";
    };
    input.addEventListener("input", show);
    show();
  }

  // 表の並べ替え（class="sortable"）
  document.querySelectorAll("table.sortable").forEach((tb) => {
    tb.querySelectorAll("th").forEach((th, col) => {
      th.addEventListener("click", () => {
        const body = tb.tBodies[0];
        const rows = [...body.rows];
        const dir = th.dataset.dir === "asc" ? "desc" : "asc";
        tb.querySelectorAll("th").forEach((x) => delete x.dataset.dir);
        th.dataset.dir = dir;
        const val = (r) => {
          const c = r.cells[col];
          const raw = c.dataset.v !== undefined ? c.dataset.v : c.textContent.trim();
          const num = parseFloat(raw.replace(/[%,]/g, ""));
          return isNaN(num) ? raw : num;
        };
        rows.sort((a, b) => {
          const x = val(a), y = val(b);
          const r = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ja");
          return dir === "asc" ? r : -r;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  });
})();
