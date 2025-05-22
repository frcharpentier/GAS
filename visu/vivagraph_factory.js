let Factory = function(){
	this.container = null;
	let graph = null;
	this.graphics =  Viva.Graph.View.svgGraphics();
	
	this.effacerGraphe = function()
	{
		if(graph)
		{
			graph.clear();
			graph = null;
		}
	};
	
	this.setContainer = function(container)
	{
		let vivaCont;
		if(graph)
		{
			graph.clear();
			graph = null;
		}
		if (container.tagName == "TR")
			vivaCont = document.createElement("td");
		else
			vivaCont = document.createElement("div");
		container.appendChild(vivaCont);
		this.container = vivaCont;
	}

	let faireNoeud = function (data)
	{
		let noeud;
		if(typeof data === 'string')
		{
			data={"svg":"cadre", "contenu":data};
		}
		else if((typeof data === "object") && (data.length == 2))
		{
			data={"svg":"cadre", "contenu":data[0], "id":data[1]};
		}
		if(data.svg == "cadre")
		{
			let rect = Viva.Graph.svg("rect");
			rect.attr("style", "fill:white;stroke-width:1;stroke:black");
			let texte = Viva.Graph.svg("text").text(data.contenu);
			noeud = Viva.Graph.svg("g");
			noeud.append(rect);
			noeud.append(texte);
			if(data.id)
			{
				noeud.attr("id", data["id"]);
			}
		}
		else
		{
			noeud = Viva.Graph.svg(data.svg);
			for(let key in data)
			{
				if (key == "svg")
				{
				}
				else if(key == "id")
				{
					noeud.attr("id", data["id"])
				}
				else if(key == "contenu")
				{
					let valeur = data[key];
					if(typeof valeur === 'string')
						noeud.text(valeur);
					else
						noeud.append(faireNoeud(valeur));
				}
				else
				{
					noeud.attr(key, data[key]);
				}
			}
		}
		return noeud;
	};
	
	this.graphics.node(function (node) {
		if(node.data)
		{
			noeud = faireNoeud(node.data);
		}
		else
		{
			noeud = Viva.Graph.svg("circle")
				.attr("r", 5)
				.attr("stroke", "black")
				.attr("fill", "lightblue");
		}
		return noeud;
	}).placeNode(function(nodeUI, pos){
		if(!("largeur" in nodeUI.node))
		{
			if (nodeUI.tagName == "g")
			{
				let rect = nodeUI.children[0];
				let texte = nodeUI.children[1];
				let bbox = texte.getBBox();
				let w = bbox.width;
				let h = bbox.height;
				rect.setAttribute("width", 1.5*w);
				rect.setAttribute("height", 1.5*h);
				rect.setAttribute("y", -1.5*h);
				
				texte.setAttribute("x", 0.25*w);
				texte.setAttribute("y", -0.5*h);
				
				nodeUI.node.largeur = 1.5*w;
				nodeUI.node.hauteur = 1.5*h;
			}
			else
			{
				let bbox = nodeUI.getBBox();
				nodeUI.node.largeur = bbox.width;
				nodeUI.node.hauteur = bbox.height;
			}
		}
		if(nodeUI.tagName == "circle")
			nodeUI.attr("cx", pos.x).attr("cy", pos.y);
		else if (nodeUI.tagName == "g")
		{
			let px, py;
			if("largeur" in nodeUI.node)
			{
				px = pos.x-nodeUI.node.largeur/2;
				py = pos.y+nodeUI.node.hauteur/2;
			}
			else
			{
				px = pos.x;
				py = pos.y;
			}
			nodeUI.attr("transform", `translate(${px},${py})`)
		}
		else
			nodeUI.attr("x", pos.x).attr("y", pos.y);
	});
	
	this.createMarker = function(id) {
		return Viva.Graph.svg('marker')
		   .attr('id', id)
		   .attr('viewBox', "0 0 15 15")
		   .attr('refX', "15")
		   .attr('refY', "7.5")
		   .attr('markerUnits', "strokeWidth")
		   .attr('markerWidth', "15")
		   .attr('markerHeight', "7.5")
		   .attr('orient', "auto");
	};
	
	this.marker = this.createMarker('Fleche');
	this.marker.append('path').attr('d', 'M 0 0 L 15 7.5 L 0 15 z');
	// Marker should be defined only once in <defs> child element of root <svg> element:
	this.defs = this.graphics.getSvgRoot().append('defs');
	this.defs.append(this.marker);

	let geom = Viva.Graph.geom();
	
	this.graphics.link(function(link){
		// L'attribut "marker-end" sert à terminer le trait par une flèche.
		if (link.data)
		{
			data = link.data;
			if (! data.bidir)
				data["bidir"] = false;
			if (! data.stroke)
				data["stroke"] = "gray";
		}
		else
			data = {"stroke": "gray", "bidir": false};
		
		let ligne =  Viva.Graph.svg('path');
		
		for (let key in data)
		{
			if (key == "bidir")
			{
			}
			else
			{
				ligne.attr(key, data[key]);
			}
		}
		if (! data.bidir)
			ligne.attr('marker-end', 'url(#Fleche)');
		
		ligne.fromId = link.fromId;
		ligne.toId = link.toId;
		return ligne
	}).placeLink(function(linkUI, pos1, pos2){
		let to, from;
		let noeud1 = graph.getNode(linkUI.fromId);
		if("largeur" in noeud1)
		{
			from = geom.intersectRect(
                        // rectangle:
                                pos1.x - noeud1.largeur/2, // left
                                pos1.y - noeud1.hauteur/2, // top
                                pos1.x + noeud1.largeur/2, // right
                                pos1.y + noeud1.hauteur/2, // bottom
                        // segment:
                                pos1.x, pos1.y, pos2.x, pos2.y)
                           || pos1; // if no intersection found - return center of the node
		}
		else
			from = pos1;
		let noeud2 = graph.getNode(linkUI.toId);
		if("largeur" in noeud1)
		{
			to = geom.intersectRect(
                        // rectangle:
                                pos2.x - noeud2.largeur/2, // left
                                pos2.y - noeud2.hauteur/2, // top
                                pos2.x + noeud2.largeur/2, // right
                                pos2.y + noeud2.hauteur/2, // bottom
                        // segment:
                                pos2.x, pos2.y, pos1.x, pos1.y)
                            || pos2; // if no intersection found - return center of the node
		}
		else
			to = pos2;
		let data = 'M' + from.x + ',' + from.y +
                           ' L' + to.x + ',' + to.y;

                linkUI.attr("d", data);
	});
	
	
	
	this.dessinerGraphe = function(G, www=-1, hhh=-1){
		if(graph)
		{
			graph.clear();
			graph = null;
		}
		graph = Viva.Graph.graph();
		let layout = Viva.Graph.Layout.forceDirected(graph, {
			springLength : 300,
			/*springCoeff : 0.0005,
			dragCoeff : 0.02,
			gravity : -1.2*/
		});	
		let renderer = Viva.Graph.View.renderer(graph, {
		container: this.container,
		layout : layout,
		graphics: this.graphics});
		
		//let i = 0;
		for (let i in G.sommets)
		{
			let nd = G.sommets[i];
			if(nd)
			{
				graph.addNode(i, nd);
				i += 1;
			}
		}
		for (let ar of G.aretes)
		{
			graph.addLink(...ar);
		}
		if(www > 0)
			this.container.style.width = www + "px";
		if(hhh > 0)
			this.container.style.height = hhh + "px";
		renderer.run();
		let hh = this.container.offsetHeight - 5;
		let ww = this.container.offsetWidth - 5;
		//console.log("(W, H) = (" + ww + ", " + hh + ")");
		let lesvg = this.graphics.getSvgRoot()
		lesvg.setAttribute('width', "" + ww + "px");
		lesvg.setAttribute('height', "" + hh + "px");
		setTimeout(function(){
			renderer.pause();
		}, 10000);
	};
};